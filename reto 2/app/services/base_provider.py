"""Abstract provider interface (Strategy pattern).

Any concrete provider must satisfy this contract so the business logic can
stay agnostic about the underlying model service.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class AIProvider(ABC):
    """Abstract base class for text-summarization providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable provider name, used in logs and responses."""

    @abstractmethod
    def summarize(self, text: str, *, system_prompt: str | None = None) -> str:
        """Return an abstractive summary of ``text``.

        Args:
            text: The input text to summarize (must be non-empty).
            system_prompt: Optional system prompt overriding the default.

        Returns:
            The generated summary as a string.

        Raises:
            app.core.exceptions.SummarizationError: on any provider failure.
        """

    @property
    @abstractmethod
    def model(self) -> str:
        """Identifier of the model this provider instance calls."""


AIProviderConfig = dict[str, Any]  # Compatibility alias kept for clarity.