from __future__ import annotations

from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from backend.config import settings


class EmbeddingService:
    """Loads the embedding model once and provides batch/query encoding."""

    def __init__(self) -> None:
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL, device="cpu")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.asarray(embeddings)
        return embeddings.astype(float).tolist()

    def embed_query(self, text: str) -> List[float]:
        if not text or not text.strip():
            raise ValueError("Query text is empty.")
        return self.embed_texts([text.strip()])[0]
