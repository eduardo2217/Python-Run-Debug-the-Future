"""Domain exceptions for the summarization service.

These map cleanly onto HTTP status codes in the API layer and onto
non-zero exit codes in the CLI layer, so callers never have to parse
raw ``openai``/``requests``/``tenacity`` exceptions.
"""

from __future__ import annotations


class SummarizationError(Exception):
    """Base exception for all application-level failures."""

    exit_code = 1
    status_code = 500

    def __init__(self, message: str, *, cause: Exception | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.cause = cause


class InvalidInputError(SummarizationError):
    """Raised when the caller provides invalid/empty text."""

    status_code = 422
    exit_code = 2


class ProviderUnavailableError(SummarizationError):
    """Raised when the AI provider cannot be reached or times out."""

    status_code = 503
    exit_code = 3


class ProviderQuotaError(SummarizationError):
    """Raised when the provider is rate limited or out of quota."""

    status_code = 429
    exit_code = 4


class ProviderResponseError(SummarizationError):
    """Raised when the provider returns a malformed/unparseable payload."""

    status_code = 502
    exit_code = 5


class RateLimitExceededError(SummarizationError):
    """Raised when the local rate limiter rejects a request."""

    status_code = 429
    exit_code = 6