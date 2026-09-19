from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def _get_env(name: str, default: str = "") -> str:
    value = os.getenv(name, default)
    return value.strip() if value is not None else default


class Settings:
    GEMINI_API_KEY: str = _get_env("GEMINI_API_KEY")
    GEMINI_MODEL: str = _get_env("GEMINI_MODEL", "gemini-3.6-flash")
    CHROMA_PATH: str = _get_env("CHROMA_PATH", str(BASE_DIR / "data" / "chroma"))
    UPLOAD_PATH: str = _get_env("UPLOAD_PATH", str(BASE_DIR / "data" / "uploads"))
    EMBEDDING_MODEL: str = _get_env("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    DEFAULT_TOP_K: int = int(_get_env("DEFAULT_TOP_K", "4"))
    CANDIDATE_TOP_K: int = int(_get_env("CANDIDATE_TOP_K", "20"))
    FINAL_TOP_K: int = int(_get_env("FINAL_TOP_K", "4"))
    RERANKER_MODEL: str = _get_env("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    RERANKER_THRESHOLD: float = float(_get_env("RERANKER_THRESHOLD", "0.0"))
    MAX_CONTEXT_CHARS: int = int(_get_env("MAX_CONTEXT_CHARS", "12000"))
    CHUNK_SIZE: int = int(_get_env("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP: int = int(_get_env("CHUNK_OVERLAP", "150"))
    MAX_UPLOAD_SIZE_MB: int = int(_get_env("MAX_UPLOAD_SIZE_MB", "20"))
    MAX_QUERY_LENGTH: int = int(_get_env("MAX_QUERY_LENGTH", "500"))


settings = Settings()
