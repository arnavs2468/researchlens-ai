"""Embedding service using local SentenceTransformers models."""

from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from app.config import settings


class EmbeddingService:
    """Singleton service for generating local vector embeddings."""

    _instance: Optional["EmbeddingService"] = None

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.embedding_model_name
        self._model: Optional[SentenceTransformer] = None

    @classmethod
    def get_instance(cls) -> "EmbeddingService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            try:
                # Try loading with local files first for zero-network latency
                self._model = SentenceTransformer(self.model_name, local_files_only=True)
            except Exception:
                # Fallback to standard load if not in local cache yet
                self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Generate normalized vector embeddings for a list of strings."""
        if not texts:
            return []
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """Generate normalized vector embedding for a single search query."""
        embedding = self.model.encode(
            query,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embedding.tolist()


embedding_service = EmbeddingService.get_instance()
