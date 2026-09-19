from __future__ import annotations

import logging
import re
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from sentence_transformers import CrossEncoder

from backend.config import settings
from backend.query_processor import QueryProcessor

logger = logging.getLogger("pdf_rag_retrieval")


class RetrievalService:
    def __init__(self, vector_store: Any, embedding_service: Any) -> None:
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.query_processor = QueryProcessor(max_length=settings.MAX_QUERY_LENGTH)
        self.reranker = None
        self._load_reranker()

    def _load_reranker(self) -> None:
        try:
            self.reranker = CrossEncoder(settings.RERANKER_MODEL, max_length=512)
        except Exception as exc:  # pragma: no cover
            logger.exception("Failed to load reranker model: %s", settings.RERANKER_MODEL)
            raise RuntimeError(f"Unable to load the reranker model: {settings.RERANKER_MODEL}") from exc

    def preprocess_query(self, query: str) -> Tuple[str, List[str]]:
        return self.query_processor.process(query)

    def dense_retrieve(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        embedding = self.embedding_service.embed_query(query)
        results = self.vector_store.query(embedding, top_k)
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        retrieved: List[Dict[str, Any]] = []
        for index, text in enumerate(documents):
            metadata = metadatas[index] if index < len(metadatas) else {}
            distance = float(distances[index]) if index < len(distances) else 0.0
            retrieved.append(
                {
                    "chunk_id": metadata.get("chunk_id", ""),
                    "filename": metadata.get("filename", "unknown.pdf"),
                    "page": metadata.get("page"),
                    "text": text,
                    "dense_distance": round(distance, 6),
                    "dense_similarity": round(1.0 - distance, 6),
                    "reranker_score": None,
                }
            )
        return retrieved

    def merge_candidates(self, candidate_lists: Sequence[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        merged: Dict[str, Dict[str, Any]] = {}
        for candidate_list in candidate_lists:
            for chunk in candidate_list:
                chunk_id = chunk.get("chunk_id")
                if not chunk_id:
                    continue
                existing = merged.get(chunk_id)
                if existing is None or chunk["dense_similarity"] > existing["dense_similarity"]:
                    merged[chunk_id] = chunk
        return sorted(merged.values(), key=lambda item: item["dense_similarity"], reverse=True)

    def rerank_candidates(self, original_query: str, candidates: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not candidates or self.reranker is None:
            return list(candidates)

        pairs = [(original_query, chunk["text"]) for chunk in candidates]
        scores = self.reranker.predict(pairs)
        reranked: List[Dict[str, Any]] = []
        for candidate, score in zip(candidates, scores):
            candidate_copy = dict(candidate)
            candidate_copy["reranker_score"] = float(score)
            reranked.append(candidate_copy)
        reranked.sort(
            key=lambda item: (
                float(item.get("reranker_score", float("-inf"))),
                float(item.get("dense_similarity", 0.0)),
            ),
            reverse=True,
        )
        return reranked

    def normalize_text(self, value: str) -> str:
        return re.sub(r"\s+", " ", value or "").strip().lower()

    def _is_near_duplicate(self, text_a: str, text_b: str) -> bool:
        if not text_a or not text_b:
            return False
        a_norm = self.normalize_text(text_a)
        b_norm = self.normalize_text(text_b)
        if a_norm == b_norm:
            return True
        if len(a_norm) < 40 or len(b_norm) < 40:
            return False
        a_tokens = set(a_norm.split())
        b_tokens = set(b_norm.split())
        if not a_tokens or not b_tokens:
            return False
        overlap = len(a_tokens & b_tokens) / max(len(a_tokens | b_tokens), 1)
        return overlap >= 0.8

    def post_process_results(self, candidates: Sequence[Dict[str, Any]], threshold: float = None) -> List[Dict[str, Any]]:
        threshold = threshold if threshold is not None else settings.RERANKER_THRESHOLD
        filtered: List[Dict[str, Any]] = []
        for candidate in candidates:
            if candidate.get("chunk_id") is None:
                continue
            if candidate.get("reranker_score") is not None and float(candidate["reranker_score"]) < float(threshold):
                continue
            filtered.append(candidate)

        deduped: List[Dict[str, Any]] = []
        for candidate in filtered:
            duplicate = False
            for existing in deduped:
                if candidate.get("chunk_id") == existing.get("chunk_id"):
                    duplicate = True
                    break
                if candidate.get("filename") == existing.get("filename") and self._is_near_duplicate(candidate.get("text", ""), existing.get("text", "")):
                    duplicate = True
                    break
            if not duplicate:
                deduped.append(candidate)

        if not deduped:
            return []

        deduped.sort(
            key=lambda item: (
                float(item.get("reranker_score", item.get("dense_similarity", 0.0))),
                float(item.get("dense_similarity", 0.0)),
            ),
            reverse=True,
        )
        return deduped

    def final_context(self, retrieved: Sequence[Dict[str, Any]], final_top_k: int = None, max_context_chars: int = None) -> List[Dict[str, Any]]:
        final_top_k = final_top_k if final_top_k is not None else settings.FINAL_TOP_K
        max_context_chars = max_context_chars if max_context_chars is not None else settings.MAX_CONTEXT_CHARS

        selected: List[Dict[str, Any]] = []
        seen_files: set[str] = set()
        for chunk in retrieved:
            if len(selected) >= final_top_k:
                break
            filename = chunk.get("filename") or "unknown.pdf"
            if filename in seen_files and len(selected) >= final_top_k:
                break
            if filename not in seen_files:
                seen_files.add(filename)
            selected.append(dict(chunk))

        context_chars = sum(len(chunk.get("text", "")) for chunk in selected)
        if context_chars > max_context_chars:
            reduced: List[Dict[str, Any]] = []
            total_chars = 0
            for chunk in selected:
                text_length = len(chunk.get("text", ""))
                if total_chars + text_length > max_context_chars:
                    continue
                reduced.append(dict(chunk))
                total_chars += text_length
            selected = reduced

        return selected[:final_top_k]

    def retrieve(self, query: str, top_k: int = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if not query or not str(query).strip():
            raise ValueError("Query cannot be empty.")

        original_query, retrieval_queries = self.preprocess_query(query)
        requested_top_k = top_k if top_k is not None else settings.DEFAULT_TOP_K
        candidate_top_k = settings.CANDIDATE_TOP_K

        candidate_lists = [self.dense_retrieve(item, candidate_top_k) for item in retrieval_queries]
        merged_candidates = self.merge_candidates(candidate_lists)
        reranked_candidates = self.rerank_candidates(original_query, merged_candidates)
        post_processed = self.post_process_results(reranked_candidates)
        final_chunks = self.final_context(post_processed, final_top_k=requested_top_k)

        retrieval_meta = {
            "original_query": original_query,
            "retrieval_queries": retrieval_queries,
            "candidate_count": len(merged_candidates),
            "final_count": len(final_chunks),
        }
        return final_chunks, retrieval_meta
