"""FastAPI application for text vectorization and duplicate detection."""

from __future__ import annotations

from fastapi import FastAPI

from src.models import (
    EmbedBatchRequest,
    EmbedBatchResponse,
    EmbedRequest,
    EmbedResponse,
    SimilarityRequest,
    SimilarityResponse,
)

app = FastAPI(
    title="Text Duplicate Finder",
    description="HTTP service for text vectorization and duplicate news detection",
    version="0.1.0",
)


@app.post("/embed", response_model=EmbedResponse)
def embed_text(request: EmbedRequest) -> EmbedResponse:
    """
    Vectorize a single text into embedding representation.

    Args:
        request: Request containing text to vectorize

    Returns:
        EmbedResponse with embedding vector and dimension

    Note:
        Currently returns a static stub vector instead of real embeddings.
        Real implementation will use BAAI/bge-large-en-v1.5 model.
    """
    # Static stub: return 1024-dimensional zero vector
    # BAAI/bge-large-en-v1.5 produces 1024-dimensional embeddings
    # TODO: Replace with actual embedding generation using request.text
    _ = request  # Acknowledge parameter (will be used when implementing real embeddings)
    dimension = 1024
    stub_embedding = [0.0] * dimension

    return EmbedResponse(embedding=stub_embedding, dimension=dimension)


@app.post("/embed/batch", response_model=EmbedBatchResponse)
def embed_batch(request: EmbedBatchRequest) -> EmbedBatchResponse:
    """
    Vectorize multiple texts into embedding representations.

    Args:
        request: Request containing list of texts to vectorize

    Returns:
        EmbedBatchResponse with list of embedding vectors, dimension, and count

    Note:
        Currently returns static stub vectors instead of real embeddings.
        Real implementation will use BAAI/bge-large-en-v1.5 model with batching.
    """
    # Static stub: return 1024-dimensional zero vectors for each text
    # BAAI/bge-large-en-v1.5 produces 1024-dimensional embeddings
    # TODO: Replace with actual batch embedding generation using request.texts
    _ = request  # Acknowledge parameter (will be used when implementing real embeddings)
    dimension = 1024
    stub_embeddings = [[0.0] * dimension for _ in request.texts]

    return EmbedBatchResponse(embeddings=stub_embeddings, dimension=dimension, count=len(request.texts))


def calculate_cosine_similarity(vector1: list[float], vector2: list[float]) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
        vector1: First embedding vector
        vector2: Second embedding vector (must be same length as vector1)

    Returns:
        Cosine similarity value between -1 and 1
        - 1.0 = identical vectors
        - 0.0 = orthogonal vectors
        - -1.0 = opposite vectors

    Note:
        Assumes vectors are already normalized. If not normalized,
        the result is still a valid cosine similarity.
    """
    # Calculate dot product
    dot_product = sum(a * b for a, b in zip(vector1, vector2, strict=True))

    # Calculate magnitudes
    magnitude1 = sum(a * a for a in vector1) ** 0.5
    magnitude2 = sum(b * b for b in vector2) ** 0.5

    # Avoid division by zero
    if magnitude1 == 0.0 or magnitude2 == 0.0:
        return 0.0

    # Calculate cosine similarity
    return float(dot_product / (magnitude1 * magnitude2))


@app.post("/similarity", response_model=SimilarityResponse)
def calculate_similarity(request: SimilarityRequest) -> SimilarityResponse:
    """
    Calculate cosine similarity between two embedding vectors.

    Args:
        request: Request containing two vectors to compare

    Returns:
        SimilarityResponse with similarity score, duplicate flag, and threshold

    Note:
        Threshold for duplicate detection is 0.85 (cosine similarity >= 0.85).
        This value is recommended for BAAI/bge-large-en-v1.5 embeddings.
    """
    threshold = 0.85
    similarity = calculate_cosine_similarity(request.vector1, request.vector2)
    is_duplicate = similarity >= threshold

    return SimilarityResponse(similarity=similarity, is_duplicate=is_duplicate, threshold=threshold)
