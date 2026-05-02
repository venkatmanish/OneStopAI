from __future__ import annotations

import hashlib
import math
from collections import Counter

from core.settings import get_settings


class EmbeddingModel:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.settings.embedding_model)
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        try:
            vectors = self.model.encode(texts, normalize_embeddings=True).tolist()
            return vectors
        except Exception:
            return [self._hash_embedding(text) for text in texts]

    @staticmethod
    def _hash_embedding(text: str, dims: int = 384) -> list[float]:
        counts: Counter[int] = Counter()
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode()).digest()
            idx = int.from_bytes(digest[:2], "big") % dims
            counts[idx] += 1
        vector = [0.0] * dims
        for idx, count in counts.items():
            vector[idx] = float(count)
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]
