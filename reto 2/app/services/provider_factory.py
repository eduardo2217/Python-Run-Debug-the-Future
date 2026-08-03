"""Provider factory (Factory pattern).

Decouples provider instantiation from usage. The factory reads the configured
provider name and builds the appropriate :class:`AIProvider` concrete class,
hiding construction details (base URL, API key, tuning knobs) from callers.
"""

from __future__ import annotations

import logging

from app.config import Settings
from app.core.exceptions import SummarizationError
from app.services.base_provider import AIProvider
from app.services.openai_provider import OpenAIProvider

logger = logging.getLogger(__name__)

# Provider name -> concrete class. Only ``openai``/``openrouter`` are wired
# today; both are served by the same OpenAI-compatible implementation.
_PROVIDERS: dict[str, type[AIProvider]] = {
    "openai": OpenAIProvider,
    "openrouter": OpenAIProvider,
}


class UnknownProviderError(SummarizationError):
    """Raised when the configured provider name is not registered."""

    status_code = 500
    exit_code = 7


def build_provider(settings: Settings) -> AIProvider:
    """Instantiate the :class:`AIProvider` matching ``settings.provider``.

    Args:
        settings: Fully-loaded application settings.

    Returns:
        A configured, ready-to-use provider instance.

    Raises:
        ProviderFactoryError: if the provider name is unknown.
    """
    provider_cls = _PROVIDERS.get(settings.provider.lower())
    if provider_cls is None:
        raise UnknownProviderError(
            f"Unknown provider '{settings.provider}'. "
            f"Available providers: {sorted(_PROVIDERS)}"
        )

    # OpenRouter free routing tolerates a missing key; the CLI/API must make
    # sure a real key is present when talking to the paid OpenAI endpoint.
    logger.info(
        "Building provider=%s model=%s base_url=%s",
        settings.provider,
        settings.model,
        settings.ai_base_url,
    )
    return provider_cls(
        model=settings.model,
        base_url=settings.ai_base_url,
        api_key=settings.api_key,
        timeout_seconds=settings.ai_timeout_seconds,
        max_tokens=settings.summary_max_tokens,
        temperature=settings.summary_temperature,
        retries=settings.tenacity_retries,
        backoff_seconds=settings.tenacity_backoff_seconds,
    )