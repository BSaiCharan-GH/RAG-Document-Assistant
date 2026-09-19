from __future__ import annotations

from typing import Any, Dict, List, TypedDict


class AgentState(TypedDict):
    question: str
    top_k: int
    messages: List[Any]
    source_mode: str
    source_reason_code: str
    source_explanation: str
    retrieved_chunks: List[Dict[str, Any]]
    web_sources: List[Dict[str, Any]]
    retrieval_meta: Dict[str, Any]
    answer: str
    iterations: int
    status: str
    tool_history: List[str]
