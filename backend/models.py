from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "PDF RAG API"


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(default=4, ge=1, le=20)


class RetrievalSummary(BaseModel):
    candidate_count: int
    final_count: int
    retrieval_queries: List[str]


class RetrievedChunk(BaseModel):
    chunk_id: str
    filename: str
    page: Optional[int] = None
    dense_distance: float = 0.0
    dense_similarity: float = 0.0
    reranker_score: Optional[float] = None
    text: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    retrieval: RetrievalSummary
    retrieved_chunks: List[RetrievedChunk]


class DocumentSummary(BaseModel):
    document_id: str
    filename: str
    chunks_count: int
    pages: List[int]
    status: str = "indexed"


class DocumentListResponse(BaseModel):
    documents: List[DocumentSummary]


class UploadResponse(BaseModel):
    status: str = "success"
    filename: str
    document_id: str
    pages: int
    chunks_created: int


class ErrorResponse(BaseModel):
    detail: str
