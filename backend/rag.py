from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

from google import genai
from google.genai import types

from backend.config import settings

logger = logging.getLogger("pdf_rag")


class RAGService:
    def __init__(self, vector_store: Any, embedding_service: Any) -> None:
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.client = None
        if settings.GEMINI_API_KEY:
            self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
        else:
            logger.warning("GEMINI_API_KEY is not set. App will start but Gemini queries will fail until it is configured.")

    def retrieve_context(self, query: str, top_k: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        query_embedding = self.embedding_service.embed_query(query)
        results = self.vector_store.query(query_embedding, top_k)

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        retrieved = []
        for idx, doc in enumerate(documents):
            metadata = metadatas[idx] if idx < len(metadatas) else {}
            retrieved.append(
                {
                    "chunk_id": metadata.get("chunk_id"),
                    "filename": metadata.get("filename"),
                    "page": metadata.get("page"),
                    "distance": round(float(distances[idx]), 4) if idx < len(distances) else 0.0,
                    "text": doc,
                }
            )

        return retrieved, results

    def generate_answer(self, question: str, context_chunks: List[str]) -> str:
        if not context_chunks:
            return "The answer is not available in the provided document."
        if not self.client:
            raise RuntimeError("GEMINI_API_KEY is not configured. Add it to the .env file before querying the RAG service.")

        context = "\n\n---\n\n".join(context_chunks)
        prompt = (
            "You are a document question-answering system. Answer the question using only the "
            "supplied retrieved context. Do not use information that is not supported by the context. "
            "If the context does not contain enough information to answer the question, state that the answer "
            "is not available in the provided document.\n\n"
            f"Question: {question}\n\nRetrieved context:\n{context}"
        )

        response = self.client.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=1024,
            ),
        )
        return (getattr(response, "text", None) or "").strip() or "The answer is not available in the provided document."
