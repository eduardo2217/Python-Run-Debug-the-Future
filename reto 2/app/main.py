"""FastAPI application wiring: providers, cache, rate limiting, routes.

Dependency injection is used throughout so every collaborator (provider,
settings, cache, limiter) can be swapped or mocked in tests.
"""

from __future__ import annotations

import hashlib
import logging

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from app.config import Settings, get_settings
from app.core.cache import TTLCache
from app.core.exceptions import RateLimitExceededError, SummarizationError
from app.core.rate_limiter import SlidingWindowRateLimiter
from app.models.schemas import HealthResponse, SummarizeRequest, SummaryResponse
from app.services.base_provider import AIProvider
from app.services.provider_factory import build_provider

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("app.main")

# Decorator hint: FastAPI Depends() below is the Dependency Injection wiring.
settings = get_settings()


def get_ai_provider(s: Settings = Depends(get_settings)) -> AIProvider:
    """Provide a shared :class:`AIProvider` instance (constructed once)."""
    return build_provider(s)


def get_cache(s: Settings = Depends(get_settings)) -> TTLCache[str, str]:
    """Provide a module-level shared cache instance."""
    if not hasattr(get_cache, "_instance"):
        get_cache._instance = TTLCache(ttl_seconds=s.cache_ttl_seconds)
    return get_cache._instance


def get_limiter(s: Settings = Depends(get_settings)) -> SlidingWindowRateLimiter:
    """Provide a module-level shared rate limiter instance."""
    if not hasattr(get_limiter, "_instance"):
        get_limiter._instance = SlidingWindowRateLimiter(
            max_requests=s.rate_limit_requests,
            window_seconds=s.rate_limit_window_seconds,
        )
    return get_limiter._instance


def _cache_key(text: str) -> str:
    """Deterministic cache key from the normalized request input."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


app = FastAPI(
    title="BBC News Summarizer",
    description=(
        "FastAPI service that summarises BBC News articles via an "
        "OpenAI-compatible (OpenRouter) language model. Includes caching, "
        "rate limiting, retries and full error mapping."
    ),
    version="1.0.0",
)


@app.exception_handler(SummarizationError)
async def handle_domain_error(_: Request, exc: SummarizationError) -> JSONResponse:
    """Map every domain exception to a clean JSON error + HTTP status."""
    logger.error("Request failed: %s", exc.message)
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.message, "type": type(exc).__name__},
    )


@app.get("/health", response_model=HealthResponse, tags=["meta"])
async def health(provider: AIProvider = Depends(get_ai_provider)) -> HealthResponse:
    """Report service status and the active model."""
    return HealthResponse(status="ok", provider=provider.name, model=provider.model)


@app.post("/summarize", response_model=SummaryResponse, tags=["summarization"])
async def summarize(
    payload: SummarizeRequest,
    request: Request,
    provider: AIProvider = Depends(get_ai_provider),
    cache: TTLCache[str, str] = Depends(get_cache),
    limiter: SlidingWindowRateLimiter = Depends(get_limiter),
    s: Settings = Depends(get_settings),
) -> SummaryResponse:
    """Summarize a news article (cached + rate limited)."""

    # Guard against trivial/oversized payloads before touching the API.
    if s.input_max_words and len(payload.text.split()) > s.input_max_words:
        from app.core.exceptions import InvalidInputError

        raise InvalidInputError(
            f"Input exceeds the {s.input_max_words}-word limit. Truncate the text."
        )

    if s.rate_limit_enabled:
        client_ip = request.client.host if request.client else "unknown"
        if not limiter.allow(client_ip):
            raise RateLimitExceededError(
                f"Rate limit reached: {s.rate_limit_requests} requests per "
                f"{s.rate_limit_window_seconds}s. Try again later."
            )

    key = _cache_key(payload.text)
    if s.cache_enabled:
        cached = cache.get(key)
        if cached is not None:
            logger.info("Cache hit for request key=%s", key[:8])
            return SummaryResponse(
                summary=cached,
                model=provider.model,
                provider=provider.name,
                cached=True,
            )

    # The system prompt is fixed server-side (see DEFAULT_SYSTEM_PROMPT).
    summary = provider.summarize(payload.text)

    if s.cache_enabled:
        cache.set(key, summary)

    return SummaryResponse(
        summary=summary, model=provider.model, provider=provider.name, cached=False
    )