from __future__ import annotations

import re
from typing import List, Tuple

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "has", "have", "he", "her", "his", "how", "i", "if", "in", "is", "it",
    "its", "of", "on", "or", "that", "the", "their", "them", "they", "this",
    "to", "was", "we", "what", "when", "where", "which", "who", "why", "will",
    "with", "you", "your"
}


class QueryProcessor:
    """Normalizes and lightly expands user queries for retrieval."""

    def __init__(self, max_length: int = 500) -> None:
        self.max_length = max_length

    @staticmethod
    def normalize_whitespace(query: str) -> str:
        return re.sub(r"\s+", " ", query or "").strip()

    @staticmethod
    def clean_query(query: str) -> str:
        cleaned = query.strip()
        cleaned = cleaned.replace("\r", " ").replace("\n", " ")
        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = re.sub(r"\s+([?.!,;:])", r"\1", cleaned)
        return cleaned.strip()

    def validate(self, query: str) -> str:
        cleaned = self.clean_query(query)
        if not cleaned:
            raise ValueError("Query cannot be empty.")
        if len(cleaned) > self.max_length:
            raise ValueError(f"Query exceeds {self.max_length} characters.")
        return cleaned

    @staticmethod
    def _extract_keywords(query: str) -> List[str]:
        tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-_.]*", query.lower())
        keywords = [token for token in tokens if token not in _STOPWORDS and len(token) > 2]
        return keywords

    def generate_variants(self, original_query: str) -> List[str]:
        variants: List[str] = [original_query]
        keywords = self._extract_keywords(original_query)
        if not keywords:
            return variants

        keyword_query = " ".join(dict.fromkeys(keywords))
        if len(keyword_query.split()) >= 3 and keyword_query.lower() != original_query.lower():
            variants.append(keyword_query)

        if len(original_query.split()) >= 8:
            simplified = " ".join(dict.fromkeys(keywords[:8]))
            if simplified and simplified.lower() not in {variant.lower() for variant in variants}:
                variants.append(simplified)

        return variants[:3]

    def process(self, query: str) -> Tuple[str, List[str]]:
        original = self.validate(query)
        retrieval_queries = self.generate_variants(original)
        return original, retrieval_queries
