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


class EmbedBatchRequest(BaseModel):
    """Request model for batch text embedding."""

    texts: list[str] = Field(..., min_length=1, description="List of texts to vectorize")

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, value: list[str]) -> list[str]:
        """Validate that texts list is not empty and contains no empty texts."""
        if not value:
            raise ValueError("Texts list cannot be empty")

        for i, text in enumerate(value):
            if not text.strip():
                raise ValueError(f"Text at index {i} cannot be empty or whitespace only")

        return value


class EmbedBatchResponse(BaseModel):
    """Response model for batch text embedding."""

    embeddings: list[list[float]] = Field(..., description="List of vector representations")
    dimension: int = Field(..., description="Dimension of embedding vectors")
    count: int = Field(..., description="Number of embeddings returned")

    @field_validator("embeddings")
    @classmethod
    def check_embeddings_not_empty(cls, value: list[list[float]]) -> list[list[float]]:
        """Validate embeddings list is not empty."""
        if not value:
            raise ValueError("Embeddings list cannot be empty")
        for i, embedding in enumerate(value):
            if not embedding:
                raise ValueError(f"Embedding at index {i} cannot be empty")
        return value
