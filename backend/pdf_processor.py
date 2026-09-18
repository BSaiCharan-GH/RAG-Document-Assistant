from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import List, Tuple

from pydantic import BaseModel
from pypdf import PdfReader

from backend.config import settings


class ChunkRecord(BaseModel):
    text: str
    page: int | None
    chunk_id: str


def sanitize_filename(filename: str) -> str:
    name = Path(filename).name
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
    return safe.strip() or "upload.pdf"


def compute_file_hash(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()


def ensure_upload_dirs() -> Path:
    upload_dir = Path(settings.UPLOAD_PATH)
    upload_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir


def extract_pdf_text(file_path: str | Path) -> Tuple[str, int]:
    reader = PdfReader(str(file_path))
    pages = reader.pages
    text_parts: List[str] = []
    for page_number, page in enumerate(pages, start=1):
        page_text = page.extract_text() or ""
        if page_text:
            text_parts.append(f"\n\n[Page {page_number}]\n{page_text.strip()}")
    extracted = "\n".join(text_parts).strip()
    return extracted, len(pages)


def chunk_text(text: str, chunk_size: int = None, chunk_overlap: int = None) -> List[ChunkRecord]:
    chunk_size = chunk_size or settings.CHUNK_SIZE
    chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
    if chunk_overlap >= chunk_size:
        raise ValueError("Chunk overlap must be smaller than chunk size.")

    page_marker_pattern = re.compile(r"\[Page\s+(\d+)\]\s*")
    matches = list(page_marker_pattern.finditer(text))
    segments: List[Tuple[int, int, int | None]] = []

    if not matches:
        segments = [(0, len(text), None)]
    else:
        for idx, match in enumerate(matches):
            start = match.start()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            page_num = int(match.group(1))
            segments.append((start, end, page_num))

    chunks: List[ChunkRecord] = []
    for segment_start, segment_end, page_num in segments:
        segment_text = text[segment_start:segment_end].strip()
        if not segment_text:
            continue
        content = re.sub(r"\s+", " ", segment_text)
        while content:
            end_index = min(len(content), chunk_size)
            fragment = content[:end_index].strip()
            if fragment:
                chunks.append(ChunkRecord(text=fragment, page=page_num, chunk_id=""))
            if len(content) <= chunk_size:
                break
            content = content[chunk_size - chunk_overlap :].lstrip()

    if not chunks:
        return [ChunkRecord(text=text.strip(), page=None, chunk_id="")]

    for idx, chunk in enumerate(chunks):
        chunk.chunk_id = f"chunk_{idx}"
    return chunks
