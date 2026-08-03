"""Unit tests. All AI calls are mocked; no live network calls.

Run with:  pytest -q   (from the repo root)
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.core.cache import TTLCache
from app.core.exceptions import (
    InvalidInputError,
    ProviderQuotaError,
    ProviderResponseError,
    ProviderUnavailableError,
)
from app.core.rate_limiter import SlidingWindowRateLimiter

# --- Caching ---------------------------------------------------------------


def test_cache_round_trip_and_ttl():
    cache = TTLCache[str, str](ttl_seconds=60)
    cache.set("k", "v")
    assert cache.get("k") == "v"

    expired = TTLCache[str, str](ttl_seconds=0)
    expired.set("k", "v")
    assert expired.get("k") is None


def test_cache_eviction_and_lru():
    cache = TTLCache[str, str](ttl_seconds=10, max_size=2)
    cache.set("a", "1")
    cache.set("b", "2")
    cache.set("c", "3")  # evicts oldest ('a')
    assert cache.get("a") is None
    assert cache.get("b") == "2"


def test_cache_expiration_on_get():
    cache = TTLCache[str, str](ttl_seconds=0.05)
    cache.set("k", "v")
    time.sleep(0.1)
    assert cache.get("k") is None


def test_cache_invalidate():
    cache = TTLCache[str, str](ttl_seconds=10)
    cache.set("k", "v")
    assert cache.invalidate("k") is True
    assert cache.get("k") is None
    assert cache.invalidate("k") is False


# --- Rate limiting ----------------------------------------------------------


def test_rate_limiter_allows_within_limit():
    limiter = SlidingWindowRateLimiter(max_requests=3, window_seconds=10)
    assert limiter.allow("ip") is True
    assert limiter.allow("ip") is True
    assert limiter.allow("ip") is True
    assert limiter.allow("ip") is False  # 4th rejected


def test_rate_limiter_is_per_client():
    limiter = SlidingWindowRateLimiter(max_requests=1, window_seconds=10)
    assert limiter.allow("ip-a") is True
    assert limiter.allow("ip-b") is True  # different client unaffected


def test_rate_limiter_window_resets():
    limiter = SlidingWindowRateLimiter(max_requests=1, window_seconds=0.05)
    assert limiter.allow("ip") is True
    assert limiter.allow("ip") is False
    time.sleep(0.1)
    assert limiter.allow("ip") is True


# --- Provider error mapping (via the injected client) ------------------------


@pytest.fixture
def provider():
    from app.services.openai_provider import OpenAIProvider

    return OpenAIProvider(
        model="test-model",
        base_url="https://example.invalid/v1",
        api_key="",
        retries=1,
        backoff_seconds=0.01,
    )


def _completion(content):
    msg = MagicMock()
    msg.message.content = content
    comp = MagicMock()
    comp.choices = [msg]
    return comp


def test_provider_extracts_summary(provider):
    provider._client = MagicMock()
    provider._client.chat.completions.create.return_value = _completion("Summary text.")
    assert provider.summarize("Some article body.") == "Summary text."


def test_provider_rejects_empty_text(provider):
    with pytest.raises(InvalidInputError):
        provider.summarize("   ")


def test_provider_returns_quota_error_on_rate_limit(provider):
    from openai import RateLimitError

    provider._client = MagicMock()
    provider._client.chat.completions.create.side_effect = RateLimitError(
        "limit", response=MagicMock(), body=None
    )
    with pytest.raises(ProviderQuotaError):
        provider.summarize("some text")


def test_provider_malformed_response(provider):
    provider._client = MagicMock()
    provider._client.chat.completions.create.return_value = MagicMock(choices=[])
    with pytest.raises(ProviderResponseError):
        provider.summarize("some text")


def test_provider_unavailable_after_retries(provider):
    from openai import APIConnectionError

    provider._client = MagicMock()
    provider._client.chat.completions.create.side_effect = APIConnectionError(
        request=MagicMock()
    )
    with pytest.raises(ProviderUnavailableError):
        provider.summarize("some text")


# --- FastAPI endpoint integration (provider mocked) ---------------------------


@pytest.fixture
def client():
    import app.main as main
    from app.core.cache import TTLCache
    from app.core.rate_limiter import SlidingWindowRateLimiter

    future: dict = {}

    def fake_provider():
        mock = MagicMock()
        mock.name = "openrouter"
        mock.model = "test-model"
        mock.summarize.side_effect = (
            lambda text, system_prompt=None: "Mocked abstractive summary."
        )
        future["provider"] = mock
        return mock

    def fake_cache():
        if "cache" not in future:
            future["cache"] = TTLCache(ttl_seconds=10)
        return future["cache"]

    def fake_limiter():
        if "limiter" not in future:
            future["limiter"] = SlidingWindowRateLimiter(max_requests=100, window_seconds=10)
        return future["limiter"]

    # Override the actual dependency functions FastAPI captured at import time.
    app = main.app
    app.dependency_overrides[main.get_ai_provider] = fake_provider
    app.dependency_overrides[main.get_cache] = fake_cache
    app.dependency_overrides[main.get_limiter] = fake_limiter
    app.dependency_overrides[main.get_settings] = lambda: MagicMock(
        cache_enabled=True,
        input_max_words=9000,
        rate_limit_enabled=True,
        rate_limit_requests=100,
        rate_limit_window_seconds=10,
    )

    with TestClient(app) as c:
        yield c, future

    app.dependency_overrides.clear()


def test_health(client):
    c, _ = client
    resp = c.get("/health")
    assert resp.status_code == 200
    assert resp.json()["model"] == "test-model"


def test_summarize_success(client):
    c, future = client
    resp = c.post("/summarize", json={"text": "Article body text here."})
    assert resp.status_code == 200
    body = resp.json()
    assert body["cached"] is False
    assert "summary" in body


def test_summarize_caches_repeated_input(client):
    c, future = client
    payload = {"text": "Repeated article body for caching test."}
    # First call populates the cache (and registers the provider mock).
    resp1 = c.post("/summarize", json=payload)
    assert resp1.status_code == 200
    first = future["provider"]
    # Force the provider mock to blow up; a cache hit must not call it again.
    first.summarize.side_effect = AssertionError("must not be called again")
    second = c.post("/summarize", json=payload)
    assert second.status_code == 200
    assert second.json()["cached"] is True


def test_summarize_rejects_blank(client):
    c, _ = client
    resp = c.post("/summarize", json={"text": "   "})
    assert resp.status_code == 422


def test_openapi_docs_enabled(client):
    c, _ = client
    assert c.get("/openapi.json").status_code == 200
    assert c.get("/docs").status_code == 200