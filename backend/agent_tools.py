from __future__ import annotations

import json
from typing import Any, Dict, List

from langchain_core.tools import tool

from backend.config import settings
from backend.query_processor import QueryProcessor


def _chunk_summary(chunks: List[Dict[str, Any]], meta: Dict[str, Any]) -> str:
    if not chunks:
        return json.dumps({"chunks": [], "candidate_count": 0, "final_count": 0, "retrieval_queries": meta.get("retrieval_queries", [])})

    serialized = [
        {
            "chunk_id": chunk.get("chunk_id", ""),
            "filename": chunk.get("filename", "unknown.pdf"),
            "page": chunk.get("page"),
            "dense_similarity": chunk.get("dense_similarity"),
            "dense_distance": chunk.get("dense_distance"),
            "reranker_score": chunk.get("reranker_score"),
            "text": chunk.get("text", "")[:2000],
        }
        for chunk in chunks
    ]
    payload = {
        "chunks": serialized,
        "candidate_count": int(meta.get("candidate_count", len(serialized))),
        "final_count": int(meta.get("final_count", len(serialized))),
        "retrieval_queries": meta.get("retrieval_queries", []),
    }
    return json.dumps(payload, ensure_ascii=False)


def build_tools(retrieval_service: Any, vector_store: Any):
    @tool
    def search_documents(query: str, top_k: int = 4) -> str:
        """Search the indexed PDFs for the most relevant passages for a user query."""
        chunks, meta = retrieval_service.retrieve(query, top_k)
        return _chunk_summary(chunks, meta)

    @tool
    def search_specific_document(document_id: str, query: str, top_k: int = 4) -> str:
        """Search within one specific document using the same retrieval stack."""
        if not document_id:
            return json.dumps({"chunks": [], "candidate_count": 0, "final_count": 0, "retrieval_queries": [query]})

        payload = vector_store.collection.get(where={"document_id": document_id}, include=["documents", "metadatas", "distances"])
        docs = payload.get("documents", []) or []
        metadatas = payload.get("metadatas", []) or []
        if not docs:
            return json.dumps({"chunks": [], "candidate_count": 0, "final_count": 0, "retrieval_queries": [query]})

        query_embedding = retrieval_service.embedding_service.embed_query(query)
        results = vector_store.query(query_embedding, top_k)
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        chunks: List[Dict[str, Any]] = []
        for idx, text in enumerate(documents):
            metadata = metadatas[idx] if idx < len(metadatas) else {}
            distance = float(distances[idx]) if idx < len(distances) else 0.0
            chunks.append(
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
        meta = {"candidate_count": len(chunks), "final_count": len(chunks), "retrieval_queries": [query]}
        return _chunk_summary(chunks, meta)

    @tool
    def expand_query(query: str) -> str:
        """Generate a few retrieval variants to improve recall for underspecified queries."""
        processor = QueryProcessor(max_length=settings.MAX_QUERY_LENGTH)
        _, variants = processor.process(query)
        return json.dumps({"expanded_queries": variants}, ensure_ascii=False)

    @tool
    def get_document_info(document_id: str) -> str:
        """Get the metadata for a specific uploaded document, including chunk count and pages."""
        if not document_id:
            return json.dumps({"document_id": document_id, "filename": "", "chunks_count": 0, "pages": []})

        result = vector_store.collection.get(where={"document_id": document_id}, include=["metadatas", "documents"])
        metadatas = result.get("metadatas", []) or []
        if not metadatas:
            return json.dumps({"document_id": document_id, "filename": "", "chunks_count": 0, "pages": []})

        filename = metadatas[0].get("filename", "unknown.pdf")
        pages = sorted({int(m.get("page")) for m in metadatas if m.get("page") is not None})
        payload = {
            "document_id": document_id,
            "filename": filename,
            "chunks_count": len(metadatas),
            "pages": pages,
        }
        return json.dumps(payload, ensure_ascii=False)

    return [search_documents, search_specific_document, expand_query, get_document_info]
