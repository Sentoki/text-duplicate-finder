"""FastAPI application for text vectorization and duplicate detection."""

from __future__ import annotations

from fastapi import FastAPI

from src.models import EmbedRequest, EmbedResponse

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
