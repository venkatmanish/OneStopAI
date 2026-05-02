from __future__ import annotations

import hashlib
import re

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams

from core.settings import get_settings


class VectorStore:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.client = QdrantClient(url=self.settings.qdrant_url)
        self.collection = self.collection_name(
            self.settings.qdrant_collection,
            self.settings.embedding_model,
        )

    @staticmethod
    def collection_name(base_name: str, embedding_model: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "_", embedding_model.lower()).strip("_")
        slug = slug[:32] or "embedding"
        digest = hashlib.sha256(embedding_model.encode()).hexdigest()[:8]
        return f"{base_name}_{slug}_{digest}"

    def ensure_collection(self, vector_size: int) -> None:
        collections = self.client.get_collections().collections
        if any(collection.name == self.collection for collection in collections):
            return
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

    def upsert(self, chunk_ids: list[str], vectors: list[list[float]], payloads: list[dict]) -> None:
        if not chunk_ids:
            return
        self.ensure_collection(len(vectors[0]))
        points = [
            PointStruct(id=chunk_id, vector=vector, payload=payload)
            for chunk_id, vector, payload in zip(chunk_ids, vectors, payloads, strict=True)
        ]
        self.client.upsert(collection_name=self.collection, points=points)

    def search(self, vector: list[float], limit: int = 8) -> list[tuple[str, float]]:
        self.ensure_collection(len(vector))
        results = self.client.search(
            collection_name=self.collection,
            query_vector=vector,
            limit=limit,
            with_payload=True,
        )
        return [(str(result.id), float(result.score)) for result in results]
