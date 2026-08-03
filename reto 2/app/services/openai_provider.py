"""OpenAI-compatible provider built on the official ``openai`` SDK.

By pointing ``base_url`` at the right endpoint this single implementation can
talk to either the official OpenAI API or OpenRouter (both use the OpenAI
chat-completions wire format). Retries use tenacity exponential backoff and
every failure mode is mapped to a domain exception.
"""

from __future__ import annotations

import logging

from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    OpenAI,
    RateLimitError,
)
from tenacity import (
    RetryError,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from app.core.exceptions import (
    InvalidInputError,
    ProviderQuotaError,
    ProviderResponseError,
    ProviderUnavailableError,
)
from app.services.base_provider import AIProvider

logger = logging.getLogger(__name__)

# Fixed system prompt. Intentionally not exposed to clients: the model judges
# the register (journalistic vs technical) and writes a summary that keeps the
# reader engaged so they do not lose interest midway.
DEFAULT_SYSTEM_PROMPT = (
    "You are a senior editor who knows how to hold a reader's attention. "
    "Read the article and decide its register: if it is news/journalistic, "
    "write a crisp, lively, impactful news summary; if it is technical, write "
    "a vivid yet accurate technical overview that avoids dull jargon. In both "
    "cases the summary must be entertaining and striking enough that a reader "
    "wants to keep reading. Keep it to 2-3 sentences, open with the single "
    "most important fact, and end with the outcome or what it means. Never "
    "invent facts that are not in the text and never use bullet points."
)

# Error types that are transient and worth retrying.
_RETRYABLE_API_ERRORS = (APIConnectionError, APITimeoutError, RateLimitError)


def _is_retryable(exc: BaseException) -> bool:
    """True for transient provider errors we are allowed to retry."""
    return isinstance(exc, _RETRYABLE_API_ERRORS)


class OpenAIProvider(AIProvider):
    """Concrete provider calling an OpenAI-compatible chat completions API."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str = "",
        timeout_seconds: float = 30.0,
        max_tokens: int = 180,
        temperature: float = 0.4,
        retries: int = 3,
        backoff_seconds: float = 1.0,
    ) -> None:
        self._model = model
        self._base_url = base_url
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._retries = retries
        self._backoff_seconds = backoff_seconds

        kwargs: dict[str, object] = {
            "base_url": base_url,
            "timeout": timeout_seconds,
            # A blank key is valid for OpenRouter's free-model routing; the
            # SDK then sends no Authorization header.
            "api_key": api_key if api_key else "not-required",
        }
        try:
            self._client = OpenAI(**kwargs)  # type: ignore[arg-type]
        except Exception as exc:  # pragma: no cover - SDK config validation
            raise ProviderUnavailableError(
                f"Could not initialise provider client for {self.name}: {exc}",
                cause=exc,
            ) from exc

        logger.info("Initialised provider model=%s base_url=%s", model, base_url)

        # Build the retry decorator at runtime so the configured retry count
        # is respected (tenacity decorators are fixed at definition time).
        self._with_retry = retry(
            retry=retry_if_exception(_is_retryable),
            stop=stop_after_attempt(self._retries),
            wait=wait_exponential(
                multiplier=self._backoff_seconds,
                min=self._backoff_seconds,
                max=min(10.0, self._backoff_seconds * 5),
            ),
            reraise=True,
        )
        self._create_completion = self._with_retry(self._create_completion)

    @property
    def name(self) -> str:
        return "openai-compatible"

    @property
    def model(self) -> str:
        return self._model

    def summarize(self, text: str, *, system_prompt: str | None = None) -> str:
        """Summarize ``text`` with the configured model (with retries)."""
        text = (text or "").strip()
        if not text:
            raise InvalidInputError("Cannot summarize empty text.")

        prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        try:
            completion = self._create_completion(prompt, text)
        except RetryError as exc:
            # All retryable attempts were exhausted.
            original = getattr(exc, "last_attempt", None)
            cause = original.exception() if original else exc
            if isinstance(cause, RateLimitError):
                raise ProviderQuotaError(
                    f"Provider rate limited after retries: {cause}", cause=cause
                ) from exc
            raise ProviderUnavailableError(
                f"Provider unavailable after retries: {cause}", cause=cause
            ) from exc
        except RateLimitError as exc:
            raise ProviderQuotaError(f"Provider rate limited: {exc}", cause=exc) from exc
        except AuthenticationError as exc:
            raise ProviderQuotaError(f"Provider authentication failed: {exc}", cause=exc) from exc
        except BadRequestError as exc:
            raise InvalidInputError(f"Request rejected by provider: {exc}", cause=exc) from exc
        except APITimeoutError as exc:
            raise ProviderUnavailableError(f"Provider timed out: {exc}", cause=exc) from exc
        except (APIConnectionError, APIError) as exc:
            raise ProviderUnavailableError(f"Provider unreachable: {exc}", cause=exc) from exc
        except Exception as exc:  # pragma: no cover - defensive
            raise ProviderUnavailableError(f"Unexpected provider error: {exc}", cause=exc) from exc

        return self._extract_summary(completion)

    def _create_completion(self, prompt: str, text: str):
        """Call the chat-completions endpoint (wrapped with retries in __init__)."""
        return self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text},
            ],
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )

    def _extract_summary(self, completion) -> str:
        """Pull the summary text out of an OpenAI chat completion object."""
        try:
            content = completion.choices[0].message.content
            if not content or not content.strip():
                raise ProviderResponseError(
                    "Provider returned an empty content field."
                )
            return content.strip()
        except (IndexError, AttributeError, TypeError) as exc:
            raise ProviderResponseError(
                f"Malformed response from provider: {exc}", cause=exc
            ) from exc