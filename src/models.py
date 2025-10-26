"""Pydantic models for request and response validation."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class EmbedRequest(BaseModel):
    """Request model for text embedding."""

    text: str = Field(..., min_length=1, description="Text to vectorize")

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, value: str) -> str:
        """Validate that text is not empty or whitespace only."""
        if not value.strip():
            raise ValueError("Text cannot be empty or whitespace only")
        return value


class EmbedResponse(BaseModel):
    """Response model for text embedding."""

    embedding: list[float] = Field(..., description="Vector representation of text")
    dimension: int = Field(..., description="Dimension of embedding vector")

    @field_validator("embedding")
    @classmethod
    def check_embedding_length(cls, value: list[float]) -> list[float]:
        """Validate embedding is not empty."""
        if not value:
            raise ValueError("Embedding cannot be empty")
        return value
