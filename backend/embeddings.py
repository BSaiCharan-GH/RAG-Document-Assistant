from __future__ import annotations

from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from backend.config import settings


class EmbeddingService:
    def __init__(self) -> None:
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL, device="cpu")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        if isinstance(embeddings, np.ndarray):
            return embeddings.astype(float).tolist()
        return [float(value) for value in embeddings]

    def embed_query(self, text: str) -> List[float]:
        if not text or not text.strip():
            raise ValueError("Query text is empty.")
        embedding = self.model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embedding[0].astype(float).tolist()
