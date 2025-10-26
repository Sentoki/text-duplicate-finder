"""Tests for FastAPI endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client() -> TestClient:
    """Create FastAPI test client."""
    from src.main import app

    return TestClient(app)


def test_embed_endpoint_success(client: TestClient) -> None:
    """Test /embed endpoint returns successful response."""
    response = client.post("/embed", json={"text": "Test message"})
    assert response.status_code == 200


def test_embed_response_format(client: TestClient) -> None:
    """Test /embed endpoint returns correct response format."""
    response = client.post("/embed", json={"text": "Sample text for embedding"})
    data = response.json()

    # Check response has required fields
    assert "embedding" in data
    assert "dimension" in data

    # Check types
    assert isinstance(data["embedding"], list)
    assert isinstance(data["dimension"], int)

    # Check embedding is list of floats
    assert all(isinstance(x, (int, float)) for x in data["embedding"])

    # Check dimension matches embedding length
    assert len(data["embedding"]) == data["dimension"]


def test_embed_empty_text_validation(client: TestClient) -> None:
    """Test /embed endpoint rejects empty text."""
    response = client.post("/embed", json={"text": ""})
    assert response.status_code == 422  # Validation error


def test_embed_missing_text_field(client: TestClient) -> None:
    """Test /embed endpoint rejects request without text field."""
    response = client.post("/embed", json={})
    assert response.status_code == 422  # Validation error


def test_embed_returns_consistent_dimension(client: TestClient) -> None:
    """Test /embed endpoint returns same dimension for all requests."""
    response1 = client.post("/embed", json={"text": "First text"})
    response2 = client.post("/embed", json={"text": "Second text"})

    dim1 = response1.json()["dimension"]
    dim2 = response2.json()["dimension"]

    assert dim1 == dim2
    assert dim1 == 1024  # Expected dimension for BAAI/bge-large-en-v1.5


# Tests for /embed/batch endpoint


def test_embed_batch_endpoint_success(client: TestClient) -> None:
    """Test /embed/batch endpoint returns successful response."""
    response = client.post(
        "/embed/batch", json={"texts": ["First text", "Second text", "Third text"]}
    )
    assert response.status_code == 200


def test_embed_batch_response_format(client: TestClient) -> None:
    """Test /embed/batch endpoint returns correct response format."""
    texts = ["Text one", "Text two", "Text three"]
    response = client.post("/embed/batch", json={"texts": texts})
    data = response.json()

    # Check response has required fields
    assert "embeddings" in data
    assert "dimension" in data
    assert "count" in data

    # Check types
    assert isinstance(data["embeddings"], list)
    assert isinstance(data["dimension"], int)
    assert isinstance(data["count"], int)

    # Check count matches input
    assert data["count"] == len(texts)

    # Check embeddings is list of lists of floats
    assert len(data["embeddings"]) == len(texts)
    for embedding in data["embeddings"]:
        assert isinstance(embedding, list)
        assert all(isinstance(x, (int, float)) for x in embedding)
        assert len(embedding) == data["dimension"]


def test_embed_batch_multiple_texts(client: TestClient) -> None:
    """Test /embed/batch endpoint processes multiple texts."""
    texts = ["First", "Second", "Third", "Fourth", "Fifth"]
    response = client.post("/embed/batch", json={"texts": texts})
    data = response.json()

    assert data["count"] == 5
    assert len(data["embeddings"]) == 5


def test_embed_batch_empty_list_validation(client: TestClient) -> None:
    """Test /embed/batch endpoint rejects empty list."""
    response = client.post("/embed/batch", json={"texts": []})
    assert response.status_code == 422  # Validation error


def test_embed_batch_empty_text_in_list(client: TestClient) -> None:
    """Test /embed/batch endpoint rejects empty text in list."""
    response = client.post("/embed/batch", json={"texts": ["Valid text", "", "Another"]})
    assert response.status_code == 422  # Validation error


def test_embed_batch_consistent_dimensions(client: TestClient) -> None:
    """Test /embed/batch endpoint returns same dimension for all embeddings."""
    response = client.post("/embed/batch", json={"texts": ["First", "Second", "Third"]})
    data = response.json()

    # All embeddings should have same dimension
    dimensions = [len(emb) for emb in data["embeddings"]]
    assert len(set(dimensions)) == 1  # All dimensions are the same
    assert dimensions[0] == data["dimension"]
    assert data["dimension"] == 1024  # Expected for BAAI/bge-large-en-v1.5
