"""Embedding model wrapper with singleton pattern."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """
    Singleton wrapper for BAAI/bge-large-en-v1.5 embedding model.

    The model is loaded once on first use and stays in memory.
    Automatically detects GPU availability and falls back to CPU if needed.
    """

    _instance: EmbeddingModel | None = None
    _model: SentenceTransformer | None = None

    def __new__(cls) -> EmbeddingModel:
        """Ensure only one instance of EmbeddingModel exists (Singleton pattern)."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_model(self) -> SentenceTransformer:
        """
        Get or load the embedding model (lazy loading).

        Returns:
            SentenceTransformer: Loaded model instance

        Note:
            Model is loaded on first call and cached for subsequent calls.
            Auto-downloads from HuggingFace Hub (~1.3 GB) on first run.
        """
        if self._model is None:
            import torch  # pylint: disable=import-outside-toplevel
            from sentence_transformers import SentenceTransformer  # pylint: disable=import-outside-toplevel

            # Auto-detect device (GPU if available, otherwise CPU)
            device = "cuda" if torch.cuda.is_available() else "cpu"

            # Load model (will download on first run)
            self._model = SentenceTransformer(
                "BAAI/bge-large-en-v1.5",
                device=device,
            )

        return self._model

    def encode(self, text: str) -> list[float]:
        """
        Encode a single text into embedding vector.

        Args:
            text: Text to vectorize

        Returns:
            Normalized 1024-dimensional embedding vector

        Note:
            Vectors are normalized (L2 norm = 1) for efficient cosine similarity
            computation via dot product.
        """
        model = self.get_model()

        # Encode and normalize
        # convert_to_tensor=False returns numpy array
        # normalize_embeddings=True ensures L2 norm = 1
        embedding = model.encode(
            text,
            convert_to_tensor=False,
            normalize_embeddings=True,
        )

        # Convert numpy array to list of floats
        return embedding.tolist()

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Encode multiple texts into embedding vectors efficiently.

        Args:
            texts: List of texts to vectorize

        Returns:
            List of normalized 1024-dimensional embedding vectors

        Note:
            Batch processing is more efficient than encoding texts individually.
            Uses batching internally for optimal GPU utilization.
        """
        model = self.get_model()

        # Encode batch and normalize
        embeddings = model.encode(
            texts,
            convert_to_tensor=False,
            normalize_embeddings=True,
            batch_size=32,  # Optimal batch size for most GPUs
        )

        # Convert numpy array to list of lists
        result: list[list[float]] = embeddings.tolist()
        return result
