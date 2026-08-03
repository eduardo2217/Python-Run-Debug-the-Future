"""Centralised application configuration.

All runtime settings are read from environment variables (and an optional
``.env`` file) through a single ``pydantic-settings`` object. This keeps
secrets out of the source tree and makes every tunable knob explicit.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables / ``.env``."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # -- AI provider -------------------------------------------------------
    # Human-readable provider name used by ``provider_factory`` to decide
    # which concrete ``AIProvider`` implementation to build.
    provider: str = Field(default="openrouter", description="Name of the AI provider.")

    # Base URL of an OpenAI-compatible chat completions API. OpenRouter and
    # the official OpenAI API both speak this protocol.
    ai_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="OpenAI-compatible chat completions base URL.",
    )

    # Model identifier. OpenRouter free routing only requires any model id;
    # a key is optional for free models (a small `OPENROUTER` header is sent).
    model: str = Field(
        default="meta-llama/llama-3.3-70b-instruct:free",
        description="Model id used for summarization.",
    )

    # Secret API key. Read from the environment ONLY. Never hardcode it.
    # It can be an OpenRouter key or an OpenAI key depending on `provider`.
    api_key: str = Field(default="", description="Provider API key (env var only).")

    # HTTP referral / app metadata used by OpenRouter's free model policy.
    # Not required to make requests; harmless metadata.
    ai_http_referer: str = Field(
        default="https://github.com/", description="OpenRouter referer metadata."
    )
    ai_http_title: str = Field(
        default="BBC News Summarizer", description="OpenRouter app title metadata."
    )

    # --- Request behaviour --------------------------------------------------
    # Timeout, in seconds, enforced on every external API call.
    ai_timeout_seconds: float = Field(default=30.0, ge=1.0, le=300.0)
    # Maximum tokens the model is allowed to produce for a summary.
    summary_max_tokens: int = Field(default=180, ge=16, le=2048)
    # Sampling temperature used when calling the model.
    summary_temperature: float = Field(default=0.4, ge=0.0, le=2.0)
    # Maximum number of words an input article may contain. Longer texts are
    # truncated client-side to keep requests cheap and within token budgets.
    input_max_words: int = Field(default=800, ge=50, le=20000)

    # --- Resilience -----------------------------------------------------------
    tenacity_retries: int = Field(default=3, ge=0, le=10)
    tenacity_backoff_seconds: float = Field(default=1.0, ge=0.0)

    # --- Caching --------------------------------------------------------------
    cache_enabled: bool = Field(default=True)
    cache_ttl_seconds: int = Field(default=3600, ge=0)

    # --- Rate limiting ---------------------------------------------------------
    rate_limit_enabled: bool = Field(default=True)
    # Max requests per unique client IP within the sliding window.
    rate_limit_requests: int = Field(default=10, ge=1)
    # Sliding window length in seconds.
    rate_limit_window_seconds: int = Field(default=60, ge=1)

    @property
    def provider_display_name(self) -> str:
        """Human readable label combining provider + model, for README/logs."""
        return f"{self.provider} / {self.model}"


@lru_cache
def get_settings() -> Settings:
    """Return a cached ``Settings`` instance (memoised across requests)."""
    return Settings()