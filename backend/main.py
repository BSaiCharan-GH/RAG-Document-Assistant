from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.config import settings
from backend.embeddings import EmbeddingService
from backend.models import (
    DocumentListResponse,
    DocumentSummary,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    RetrievalSummary,
    RetrievedChunk,
    UploadResponse,
)
from backend.pdf_processor import compute_file_hash, ensure_upload_dirs, extract_pdf_text, sanitize_filename, chunk_text
from backend.rag import RAGService
from backend.retrieval import RetrievalService
from backend.vector_store import VectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("pdf_rag_service")

app = FastAPI(title="PDF RAG Assistant", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "frontend"
UPLOAD_DIR = Path(settings.UPLOAD_PATH)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

vector_store = VectorStore()
embedding_service = EmbeddingService()
retrieval_service = RetrievalService(vector_store=vector_store, embedding_service=embedding_service)
rag_service = RAGService(vector_store=vector_store, embedding_service=embedding_service, retrieval_service=retrieval_service)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", service="PDF RAG API")


@app.get("/")
def serve_frontend() -> FileResponse:
    index_file = STATIC_DIR / "index.html"
    return FileResponse(index_file)


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)) -> UploadResponse:
    logger.info("PDF upload request received: %s", file.filename)

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected.")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    file_bytes = await file.read()
    if len(file_bytes) > settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File exceeds {settings.MAX_UPLOAD_SIZE_MB} MB limit.")

    file_hash = compute_file_hash(file_bytes)
    safe_name = sanitize_filename(file.filename)
    upload_subdir = ensure_upload_dirs()
    stored_filename = f"{file_hash}.pdf"
    target_path = upload_subdir / stored_filename

    if target_path.exists() or vector_store.has_document(file_hash):
        logger.info("Duplicate PDF detected: %s", safe_name)
        raise HTTPException(status_code=409, detail="This document has already been uploaded.")

    with open(target_path, "wb") as handle:
        handle.write(file_bytes)

    try:
        extracted_text, page_count = extract_pdf_text(target_path)
        if not extracted_text.strip():
            raise HTTPException(status_code=400, detail="The uploaded PDF does not contain readable text.")

        chunks = chunk_text(extracted_text)
        chunk_texts = [chunk.text for chunk in chunks]
        chunk_metadatas = [
            {
                "document_id": file_hash,
                "filename": safe_name,
                "page": chunk.page,
                "chunk_id": f"{file_hash}_{index}",
                "source": "uploaded_pdf",
            }
            for index, chunk in enumerate(chunks)
        ]
        chunk_embeddings = embedding_service.embed_texts(chunk_texts)

        vector_store.add_document(
            file_hash, safe_name, chunk_texts, chunk_metadatas, chunk_embeddings
        )
        logger.info("Ingestion complete: %s pages=%s chunks=%s", safe_name, page_count, len(chunks))
        return UploadResponse(
            status="success",
            filename=safe_name,
            document_id=file_hash,
            pages=page_count,
            chunks_created=len(chunks),
        )
    except HTTPException:
        if target_path.exists():
            try:
                target_path.unlink()
            except OSError:
                pass
        raise
    except Exception as exc:
        logger.exception("Error processing uploaded PDF")
        if target_path.exists():
            try:
                target_path.unlink()
            except OSError:
                pass
        raise HTTPException(status_code=500, detail="Failed to process the uploaded PDF.") from exc


@app.post("/query", response_model=QueryResponse)
async def query_documents(payload: QueryRequest) -> QueryResponse:
    if not payload.query or not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    if len(payload.query.strip()) > settings.MAX_QUERY_LENGTH:
        raise HTTPException(status_code=400, detail=f"Query exceeds {settings.MAX_QUERY_LENGTH} characters.")
    if not vector_store.has_documents():
        raise HTTPException(status_code=404, detail="No indexed documents available. Upload a PDF first.")

    try:
        retrieved, retrieval_meta = rag_service.retrieve_context(payload.query, payload.top_k)
        context_texts = [item["text"] for item in retrieved]
        answer = rag_service.generate_answer(payload.query, context_texts)
        return QueryResponse(
            question=payload.query,
            answer=answer,
            retrieval=RetrievalSummary(
                candidate_count=int(retrieval_meta.get("candidate_count", len(retrieved))),
                final_count=int(retrieval_meta.get("final_count", len(retrieved))),
                retrieval_queries=[str(item) for item in retrieval_meta.get("retrieval_queries", [payload.query])],
            ),
            retrieved_chunks=[
                RetrievedChunk(
                    chunk_id=item.get("chunk_id", ""),
                    filename=item.get("filename", "unknown.pdf"),
                    page=item.get("page"),
                    dense_distance=float(item.get("dense_distance", 0.0)),
                    dense_similarity=float(item.get("dense_similarity", 1.0 - float(item.get("dense_distance", 0.0)))),
                    reranker_score=float(item["reranker_score"]) if item.get("reranker_score") is not None else None,
                    text=item.get("text", ""),
                )
                for item in retrieved
            ],
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.exception("Gemini configuration error during query")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail="Unable to process the query at the moment.") from exc


@app.get("/documents", response_model=DocumentListResponse)
def list_documents() -> DocumentListResponse:
    documents = vector_store.list_documents()
    return DocumentListResponse(documents=[DocumentSummary(**doc) for doc in documents])


@app.delete("/documents/{document_id}")
def delete_document(document_id: str) -> JSONResponse:
    success = vector_store.delete_document(document_id)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found.")

    upload_dir = Path(settings.UPLOAD_PATH)
    target_path = upload_dir / f"{document_id}.pdf"
    if target_path.exists():
        target_path.unlink()

    return JSONResponse({"status": "deleted", "document_id": document_id})



@app.exception_handler(HTTPException)
async def http_exception_handler(_, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
