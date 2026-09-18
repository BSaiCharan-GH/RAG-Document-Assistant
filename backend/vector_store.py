from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import chromadb

from backend.config import settings


class VectorStore:
    def __init__(self) -> None:
        persist_dir = Path(settings.CHROMA_PATH)
        persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.collection = self.client.get_or_create_collection(
            name="pdf_rag_collection",
            metadata={"hnsw:space": "cosine"},
        )

    def add_document(
        self,
        document_id: str,
        filename: str,
        chunks: List[str],
        metadatas: List[Dict[str, Any]],
        embeddings: List[List[float]],
    ) -> int:
        if not chunks:
            return 0

        existing_ids = self.collection.get(where={"document_id": document_id}).get("ids", [])
        if existing_ids:
            self.collection.delete(ids=existing_ids)

        chunk_ids = [f"{document_id}_{index}" for index in range(len(chunks))]
        self.collection.add(
            ids=chunk_ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        return len(chunks)

    def query(self, query_embedding: List[float], top_k: int) -> Dict[str, Any]:
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

    def list_documents(self) -> List[Dict[str, Any]]:
        raw_result = self.collection.get(include=["metadatas", "documents"])
        document_map: Dict[str, Dict[str, Any]] = {}

        for metadata, document in zip(raw_result.get("metadatas", []), raw_result.get("documents", [])):
            if not metadata:
                continue
            document_id = metadata.get("document_id")
            if not document_id:
                continue

            entry = document_map.setdefault(
                document_id,
                {
                    "document_id": document_id,
                    "filename": metadata.get("filename", "unknown.pdf"),
                    "chunks_count": 0,
                    "pages": set(),
                },
            )
            entry["chunks_count"] += 1
            page = metadata.get("page")
            if page is not None:
                entry["pages"].add(int(page))

        documents = []
        for item in document_map.values():
            documents.append(
                {
                    "document_id": item["document_id"],
                    "filename": item["filename"],
                    "chunks_count": item["chunks_count"],
                    "pages": sorted(item["pages"]),
                    "status": "indexed",
                }
            )
        return sorted(documents, key=lambda entry: entry["filename"].lower())

    def delete_document(self, document_id: str) -> bool:
        ids = self.collection.get(where={"document_id": document_id}).get("ids", [])
        if not ids:
            return False
        self.collection.delete(ids=ids)
        return True

    def has_documents(self) -> bool:
        return self.collection.count() > 0

    def has_document(self, document_id: str) -> bool:
        result = self.collection.get(
                where={"document_id": document_id},
            )
        return bool(result.get("ids", []))