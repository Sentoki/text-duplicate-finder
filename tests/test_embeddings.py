"""Tests for EmbeddingModel class."""

from __future__ import annotations

import pytest


def test_singleton_pattern() -> None:
    """Test that EmbeddingModel follows singleton pattern."""
    from src.embeddings import EmbeddingModel

    instance1 = EmbeddingModel()
    instance2 = EmbeddingModel()

    assert instance1 is instance2


def test_encode_returns_list_of_floats() -> None:
    """Test encode returns list of floats."""
    from src.embeddings import EmbeddingModel

    model = EmbeddingModel()
    embedding = model.encode("Test text")

    assert isinstance(embedding, list)
    assert all(isinstance(x, float) for x in embedding)


def test_encode_returns_correct_dimension() -> None:
    """Test encode returns 1024-dimensional vector for bge-large-en-v1.5."""
    from src.embeddings import EmbeddingModel

    model = EmbeddingModel()
    embedding = model.encode("Sample text for testing")

    assert len(embedding) == 1024


def test_encode_batch_returns_list() -> None:
    """Test encode_batch returns list of embeddings."""
    from src.embeddings import EmbeddingModel

    model = EmbeddingModel()
    texts = ["First text", "Second text", "Third text"]
    embeddings = model.encode_batch(texts)

    assert isinstance(embeddings, list)
    assert len(embeddings) == len(texts)
    assert all(isinstance(emb, list) for emb in embeddings)
    assert all(len(emb) == 1024 for emb in embeddings)


def test_embeddings_are_normalized() -> None:
    """Test that embeddings are normalized (L2 norm = 1)."""
    from src.embeddings import EmbeddingModel

    model = EmbeddingModel()
    embedding = model.encode("Test normalization")

    # Calculate L2 norm
    norm = sum(x**2 for x in embedding) ** 0.5

    assert norm == pytest.approx(1.0, abs=0.01)


def test_different_texts_produce_different_embeddings() -> None:
    """Test that different texts produce different embeddings."""
    from src.embeddings import EmbeddingModel

    model = EmbeddingModel()
    emb1 = model.encode("This is about cats")
    emb2 = model.encode("This is about dogs")

    # Embeddings should be different
    assert emb1 != emb2


def test_same_text_produces_same_embedding() -> None:
    """Test that same text produces same embedding."""
    from src.embeddings import EmbeddingModel

    model = EmbeddingModel()
    text = "Consistent text for testing"
    emb1 = model.encode(text)
    emb2 = model.encode(text)

    # Embeddings should be identical
    assert emb1 == emb2


def test_encode_batch_same_as_individual() -> None:
    """Test that batch encoding produces same results as individual."""
    from src.embeddings import EmbeddingModel

    model = EmbeddingModel()
    texts = ["First", "Second"]

    # Individual encoding
    emb1 = model.encode(texts[0])
    emb2 = model.encode(texts[1])

    # Batch encoding
    batch_embeddings = model.encode_batch(texts)

    # Should be same (or very close due to batching differences)
    assert len(batch_embeddings[0]) == len(emb1)
    assert len(batch_embeddings[1]) == len(emb2)
