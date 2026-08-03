"""Pydantic request/response schemas for the FastAPI API."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class SummarizeRequest(BaseModel):
    """Payload submitted by clients to summarize a piece of text.

    The summarization prompt is fixed server-side; clients only supply text.
    """

    text: str = Field(
        ...,
        min_length=1,
        max_length=500_000,
        description="The article text to summarize. Must be non-empty.",
    )

    @field_validator("text")
    @classmethod
    def _text_must_not_be_blank(cls, value: str) -> str:
        """Reject whitespace-only payloads with a clear 422 error."""
        if not value.strip():
            raise ValueError("text must contain at least one non-whitespace character")
        return value


class SummaryResponse(BaseModel):
    """Successful summarization result payload."""

    summary: str = Field(..., description="The generated abstractive summary.")
    model: str = Field(..., description="Model identifier that produced it.")
    provider: str = Field(..., description="Provider name that served the request.")
    cached: bool = Field(
        default=False,
        description="True when the result came from the cache, not the API.",
    )


class HealthResponse(BaseModel):
    """Health-check payload reported by the /health endpoint."""

    status: str
    provider: str
    model: str