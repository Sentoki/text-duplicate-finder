"""Pydantic models for request and response validation."""

from __future__ import annotations

from typing import Any

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


class SimilarityRequest(BaseModel):
    """Request model for similarity calculation between two vectors."""

    vector1: list[float] = Field(..., min_length=1, description="First embedding vector")
    vector2: list[float] = Field(..., min_length=1, description="Second embedding vector")

    @field_validator("vector1", "vector2")
    @classmethod
    def check_vector_not_empty(cls, value: list[float]) -> list[float]:
        """Validate that vector is not empty."""
        if not value:
            raise ValueError("Vector cannot be empty")
        return value

    @field_validator("vector2")
    @classmethod
    def check_same_dimension(cls, value: list[float], info: Any) -> list[float]:
        """Validate that vector2 has same dimension as vector1."""
        if "vector1" in info.data and len(info.data["vector1"]) != len(value):
            raise ValueError(f"Vectors must have the same dimension. Got {len(info.data['vector1'])} and {len(value)}")
        return value


class SimilarityResponse(BaseModel):
    """Response model for similarity calculation."""

    similarity: float = Field(..., ge=-1.0, le=1.0, description="Cosine similarity between vectors")
    is_duplicate: bool = Field(..., description="Whether vectors are considered duplicates")
    threshold: float = Field(..., description="Threshold used for duplicate detection")
