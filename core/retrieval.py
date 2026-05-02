from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from core.db import Chunk, session_scope
from core.embeddings import EmbeddingModel
from core.repository import Repository
from core.schemas import RetrievedChunk, SourceRef
from core.settings import get_settings


class HybridRetriever:
    DOCUMENT_TYPE_PREFIX = "__document_type__:"
    DOCUMENT_TYPE_EXTENSIONS = {
        "presentation": {".ppt", ".pptx"},
        "spreadsheet": {".csv", ".xls", ".xlsx"},
        "pdf": {".pdf"},
        "image": {".jpeg", ".jpg", ".png", ".tif", ".tiff"},
    }
    RANK_STOPWORDS = {
        "about",
        "after",
        "again",
        "also",
        "and",
        "answer",
        "are",
        "for",
        "from",
        "how",
        "into",
        "the",
        "this",
        "that",
        "those",
        "use",
        "was",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
    }

    def __init__(self) -> None:
        self.settings = get_settings()
        self.embedding_model = EmbeddingModel()
        self._reranker = None

    def retrieve(self, query: str, top_k: int = 6, target_source: str | None = None) -> list[RetrievedChunk]:
        with session_scope() as session:
            repo = Repository(session)
            chunks = repo.chunks_for_active_versions()
            if target_source:
                chunks = [chunk for chunk in chunks if self._matches_target(chunk, target_source)]
            if not chunks:
                return []
            search_query = self._search_query(query)
            expanded_queries = self._expanded_queries(search_query)
            bm25_scores = self._bm25_multi(expanded_queries, chunks)
            allowed_ids = {chunk.chunk_id for chunk in chunks}
            vector_scores = self._vector_multi(
                expanded_queries,
                allowed_ids=allowed_ids,
                limit=max(32, top_k * 6),
            )
            merged = self._merge_scores(chunks, bm25_scores, vector_scores)
            selected_ids = [chunk_id for chunk_id, _ in merged[: top_k * 6]]
            selected = {chunk.chunk_id: chunk for chunk in repo.chunks_by_ids(selected_ids)}
            reranked = self._rerank(
                search_query,
                [(selected[cid], score) for cid, score in merged if cid in selected],
            )
            reranked = self._cross_encoder_rerank(search_query, reranked[: max(top_k * 4, 12)])
            diversified = self._mmr_select(search_query, reranked, top_k=top_k)
            return [self._to_retrieved(chunk, score, search_query) for chunk, score in diversified]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def _bm25(self, query: str, chunks: list[Chunk]) -> dict[str, float]:
        tokenized = [self._tokenize(chunk.text) for chunk in chunks]
        bm25 = BM25Okapi(tokenized)
        scores = bm25.get_scores(self._tokenize(query))
        max_score = max(scores) if len(scores) else 0
        return {
            chunk.chunk_id: float(score / max_score) if max_score else 0.0
            for chunk, score in zip(chunks, scores, strict=True)
        }

    def _bm25_multi(self, queries: list[str], chunks: list[Chunk]) -> dict[str, float]:
        merged = {chunk.chunk_id: 0.0 for chunk in chunks}
        for index, query in enumerate(queries):
            weight = 1.0 if index == 0 else 0.82
            for chunk_id, score in self._bm25(query, chunks).items():
                merged[chunk_id] = max(merged.get(chunk_id, 0.0), score * weight)
        return merged

    def _vector(self, query: str, limit: int = 12) -> dict[str, float]:
        try:
            from core.vector_store import VectorStore

            vector = self.embedding_model.embed([query])[0]
            return dict(VectorStore().search(vector, limit=limit))
        except Exception:
            return {}

    def _vector_multi(
        self,
        queries: list[str],
        allowed_ids: set[str],
        limit: int,
    ) -> dict[str, float]:
        merged: dict[str, float] = {}
        for index, query in enumerate(queries):
            weight = 1.0 if index == 0 else 0.86
            for chunk_id, score in self._vector(query, limit=limit).items():
                if chunk_id not in allowed_ids:
                    continue
                merged[chunk_id] = max(merged.get(chunk_id, 0.0), score * weight)
        return merged

    @staticmethod
    def _merge_scores(
        chunks: list[Chunk], bm25_scores: dict[str, float], vector_scores: dict[str, float]
    ) -> list[tuple[str, float]]:
        chunk_ids = {chunk.chunk_id for chunk in chunks}
        merged = []
        for chunk_id in chunk_ids:
            score = 0.55 * bm25_scores.get(chunk_id, 0.0) + 0.45 * vector_scores.get(chunk_id, 0.0)
            merged.append((chunk_id, score))
        return sorted(merged, key=lambda item: item[1], reverse=True)

    def _rerank(self, query: str, candidates: list[tuple[Chunk, float]]) -> list[tuple[Chunk, float]]:
        query_terms = {
            token
            for token in self._tokenize(query)
            if len(token) >= 3 and token not in self.RANK_STOPWORDS
        }
        normalized_query = " ".join(self._tokenize(query))
        reranked = []
        for chunk, score in candidates:
            chunk_terms = set(self._tokenize(chunk.text))
            overlap = len(query_terms & chunk_terms) / max(len(query_terms), 1)
            chunk_text = " ".join(self._tokenize(chunk.text))
            parent_text = " ".join(self._tokenize((chunk.extra or {}).get("parent_context", "")))
            exact_phrase_boost = 0.12 if normalized_query and normalized_query in chunk_text else 0.0
            parent_overlap = len(query_terms & set(self._tokenize(parent_text))) / max(len(query_terms), 1)
            reranked.append(
                (
                    chunk,
                    round(score * 0.58 + overlap * 0.3 + parent_overlap * 0.08 + exact_phrase_boost, 4),
                )
            )
        return sorted(reranked, key=lambda item: item[1], reverse=True)

    def _cross_encoder_rerank(
        self,
        query: str,
        candidates: list[tuple[Chunk, float]],
    ) -> list[tuple[Chunk, float]]:
        if not self.settings.reranker_model or not candidates:
            return candidates
        try:
            reranker = self._cross_encoder
            pairs = [[query, self._rerank_text(chunk)] for chunk, _ in candidates]
            raw_scores = reranker.predict(pairs)
            normalized = self._normalize_scores([float(score) for score in raw_scores])
            rescored = [
                (chunk, round(base_score * 0.35 + rerank_score * 0.65, 4))
                for (chunk, base_score), rerank_score in zip(candidates, normalized, strict=True)
            ]
            return sorted(rescored, key=lambda item: item[1], reverse=True)
        except Exception:
            return candidates

    @property
    def _cross_encoder(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder(self.settings.reranker_model)
        return self._reranker

    @staticmethod
    def _normalize_scores(scores: list[float]) -> list[float]:
        if not scores:
            return []
        low = min(scores)
        high = max(scores)
        if high == low:
            return [1.0 for _ in scores]
        return [(score - low) / (high - low) for score in scores]

    @staticmethod
    def _rerank_text(chunk: Chunk) -> str:
        parent_context = (chunk.extra or {}).get("parent_context")
        if parent_context:
            return str(parent_context)[:3200]
        return chunk.text

    def _mmr_select(
        self,
        query: str,
        candidates: list[tuple[Chunk, float]],
        top_k: int,
    ) -> list[tuple[Chunk, float]]:
        if len(candidates) <= top_k:
            return candidates[:top_k]
        selected: list[tuple[Chunk, float]] = []
        remaining = candidates[:]
        query_terms = self._content_terms(query)
        while remaining and len(selected) < top_k:
            if not selected:
                selected.append(remaining.pop(0))
                continue
            best_index = max(
                range(len(remaining)),
                key=lambda index: self._mmr_score(
                    remaining[index],
                    selected,
                    query_terms=query_terms,
                ),
            )
            selected.append(remaining.pop(best_index))
        return selected

    @classmethod
    def _mmr_score(
        cls,
        candidate: tuple[Chunk, float],
        selected: list[tuple[Chunk, float]],
        query_terms: set[str],
    ) -> float:
        chunk, score = candidate
        candidate_terms = cls._content_terms(cls._rerank_text(chunk))
        query_overlap = len(candidate_terms & query_terms) / max(len(query_terms), 1)
        max_similarity = max(
            cls._jaccard(candidate_terms, cls._content_terms(cls._rerank_text(selected_chunk)))
            for selected_chunk, _ in selected
        )
        same_document_penalty = 0.04 if any(selected_chunk.document_id == chunk.document_id for selected_chunk, _ in selected) else 0.0
        return score * 0.72 + query_overlap * 0.18 - max_similarity * 0.22 - same_document_penalty

    @classmethod
    def _content_terms(cls, text: str) -> set[str]:
        return {
            token
            for token in cls._tokenize(text)
            if len(token) >= 3 and token not in cls.RANK_STOPWORDS
        }

    @staticmethod
    def _jaccard(left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        return len(left & right) / len(left | right)

    @staticmethod
    def _to_retrieved(chunk: Chunk, score: float, query: str) -> RetrievedChunk:
        extra = chunk.extra or {}
        excerpt_text = str(extra.get("parent_context") or chunk.text)
        return RetrievedChunk(
            chunk_id=chunk.chunk_id,
            text=chunk.text,
            score=score,
            source=SourceRef(
                document_id=chunk.document_id,
                version_id=chunk.version_id,
                file_name=extra.get("file_name"),
                source_type=chunk.source_type,
                page=chunk.page,
                sheet=chunk.sheet,
                chunk_id=chunk.chunk_id,
                score=score,
                excerpt=HybridRetriever._excerpt(excerpt_text, query),
            ),
            metadata=extra,
        )

    @classmethod
    def _matches_target(cls, chunk: Chunk, target_source: str) -> bool:
        file_name = (chunk.extra or {}).get("file_name") or ""
        document_type = cls._target_document_type(target_source)
        if document_type:
            return cls._file_matches_document_type(file_name, document_type)
        return file_name == target_source

    @classmethod
    def _target_document_type(cls, target_source: str) -> str | None:
        if target_source.startswith(cls.DOCUMENT_TYPE_PREFIX):
            document_type = target_source.removeprefix(cls.DOCUMENT_TYPE_PREFIX).strip().lower()
            return document_type or None
        return None

    @classmethod
    def _file_matches_document_type(cls, file_name: str, document_type: str) -> bool:
        suffix = "." + file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""
        return suffix in cls.DOCUMENT_TYPE_EXTENSIONS.get(document_type, set())

    @staticmethod
    def _search_query(query: str) -> str:
        term_match = re.search(r"(?im)^Term to explain:\s*(.+?)\s*$", query)
        if term_match:
            term = term_match.group(1).strip()
            return f"{term} meaning definition context"

        follow_up = ""
        if "Follow-up question:" in query:
            follow_up = query.rsplit("Follow-up question:", 1)[1].splitlines()[0].strip()

        referenced = ""
        if "The referenced numbered item from the previous answer is exactly:" in query:
            referenced = query.split(
                "The referenced numbered item from the previous answer is exactly:",
                1,
            )[1].split("Answer about that exact item", 1)[0]
            referenced = re.sub(r"\s+", " ", referenced).strip()

        assistant_matches = re.findall(
            r"Assistant(?:\[[^\]]+\])?:\s*(.*?)(?=\n(?:User|Assistant)(?:\[[^\]]+\])?:|\nFollow-up question:|\Z)",
            query,
            flags=re.DOTALL,
        )
        previous_answer = re.sub(r"\s+", " ", assistant_matches[-1]).strip() if assistant_matches else ""

        parts = [follow_up, referenced or previous_answer]
        cleaned = " ".join(part for part in parts if part).strip()
        if cleaned:
            return cleaned[:1000]
        return query

    @classmethod
    def _expanded_queries(cls, query: str) -> list[str]:
        cleaned = re.sub(r"\s+", " ", query).strip()
        if not cleaned:
            return []
        expansions = [cleaned]

        content = " ".join(sorted(cls._content_terms(cleaned)))
        if content and content != cleaned:
            expansions.append(content)

        quoted_phrases = re.findall(r'"([^"]{3,120})"', cleaned)
        expansions.extend(quoted_phrases)

        if re.search(r"\bsteps?\b", cleaned, flags=re.IGNORECASE):
            expansions.append(f"{cleaned} process sequence workflow")
        if re.search(r"\bmessage|empath(?:ic|y|etic)\b", cleaned, flags=re.IGNORECASE):
            expansions.append(f"{cleaned} communication template outreach")

        deduped: list[str] = []
        seen: set[str] = set()
        for item in expansions:
            key = item.lower()
            if key and key not in seen:
                seen.add(key)
                deduped.append(item[:1000])
        return deduped[:4]

    @classmethod
    def _excerpt(cls, text: str, query: str, window: int = 520) -> str:
        terms = [
            token
            for token in cls._tokenize(query)
            if len(token) >= 3 and token not in cls.RANK_STOPWORDS
        ]
        lowered = text.lower()
        first_hit = None
        for term in terms:
            position = lowered.find(term)
            if position >= 0 and (first_hit is None or position < first_hit):
                first_hit = position
        if first_hit is None:
            return text[:window]
        start = max(0, first_hit - window // 3)
        end = min(len(text), start + window)
        excerpt = text[start:end].strip()
        if start > 0:
            excerpt = "..." + excerpt
        if end < len(text):
            excerpt = excerpt + "..."
        return excerpt
