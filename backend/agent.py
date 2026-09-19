from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from backend.agent_state import AgentState
from backend.agent_tools import build_tools
from backend.config import settings
from backend.rag import RAGService

logger = logging.getLogger("pdf_rag_agent")


class AgenticRAGService:
    def __init__(self, vector_store: Any, embedding_service: Any, retrieval_service: Any, rag_service: RAGService | None = None) -> None:
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.retrieval_service = retrieval_service
        self.rag_service = rag_service or RAGService(vector_store, embedding_service, retrieval_service)
        self.tools = build_tools(retrieval_service, vector_store)
        self.llm = None
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            if settings.GEMINI_API_KEY:
                self.llm = ChatGoogleGenerativeAI(
                    model=settings.AGENT_MODEL,
                    temperature=settings.AGENT_TEMPERATURE,
                    api_key=settings.GEMINI_API_KEY,
                )
        except Exception as exc:  # pragma: no cover
            logger.warning("Unable to initialize agent LLM: %s", exc)
            self.llm = None
        self.graph = self.build_graph()

    def _dedupe_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        unique: Dict[str, Dict[str, Any]] = {}
        for chunk in chunks:
            chunk_id = chunk.get("chunk_id") or chunk.get("filename", "unknown") + str(chunk.get("page")) + chunk.get("text", "")[:80]
            if chunk_id not in unique:
                unique[chunk_id] = chunk
        return list(unique.values())

    def _extract_tool_results(self, messages: List[Any]) -> List[Dict[str, Any]]:
        merged: List[Dict[str, Any]] = []
        for message in reversed(messages):
            if not hasattr(message, "tool_call_id"):
                continue
            content = message.content
            if not content:
                continue
            try:
                payload = json.loads(content)
            except (TypeError, ValueError):
                continue
            if not isinstance(payload, dict):
                continue
            chunks = payload.get("chunks")
            if isinstance(chunks, list):
                merged.extend(chunks)
        return self._dedupe_chunks(merged)

    def _route_after_agent(self, state: AgentState) -> str:
        if state.get("iterations", 0) >= settings.AGENT_MAX_ITERATIONS:
            return "finalize"
        last_message = state.get("messages", [])[-1] if state.get("messages") else None
        if getattr(last_message, "tool_calls", None):
            return "tools"
        return "finalize"

    def _agent_node(self, state: AgentState) -> AgentState:
        question = state["question"]
        messages = list(state.get("messages", []))
        if not messages:
            messages = [HumanMessage(content=question)]

        if not self.llm:
            chunks, meta = self.retrieval_service.retrieve(question, state.get("top_k", settings.DEFAULT_TOP_K))
            return {
                **state,
                "retrieved_chunks": chunks,
                "retrieval_meta": meta,
                "answer": "",
                "iterations": state.get("iterations", 0) + 1,
                "status": "ready_for_finalization",
                "messages": messages,
            }

        if state.get("iterations", 0) >= settings.AGENT_MAX_ITERATIONS:
            return {**state, "status": "ready_for_finalization", "messages": messages}

        tool_calls = []
        for message in messages[::-1][:3]:
            tool_calls = getattr(message, "tool_calls", []) or tool_calls
            if tool_calls:
                break

        if not tool_calls:
            system_prompt = (
                "You are an agentic document-retrieval planner. First search the document store for the most relevant evidence. "
                "Only call a tool when it helps answer the user question. If the evidence is sufficient, answer directly without extra tool calls."
            )
            response = self.llm.bind_tools(self.tools).invoke([
                SystemMessage(content=system_prompt),
                *messages,
            ])
            return {
                **state,
                "messages": messages + [response],
                "iterations": state.get("iterations", 0) + 1,
                "status": "awaiting_tool_result" if getattr(response, "tool_calls", None) else "ready_for_finalization",
            }

        return {**state, "status": "awaiting_tool_result", "messages": messages}

    def _finalize_node(self, state: AgentState) -> AgentState:
        question = state["question"]
        chunks = state.get("retrieved_chunks", [])
        if not chunks:
            chunks = self._extract_tool_results(state.get("messages", []))
        if not chunks:
            chunks, meta = self.retrieval_service.retrieve(question, state.get("top_k", settings.DEFAULT_TOP_K))
            state["retrieved_chunks"] = chunks
            state["retrieval_meta"] = meta
        else:
            state["retrieved_chunks"] = self._dedupe_chunks(chunks)
            state["retrieval_meta"] = state.get("retrieval_meta", {"candidate_count": len(chunks), "final_count": len(chunks), "retrieval_queries": [question]})

        context = [chunk.get("text", "") for chunk in state["retrieved_chunks"] if chunk.get("text")]
        answer = self.rag_service.generate_answer(question, context)
        state["answer"] = answer
        state["status"] = "completed"
        return state

    def build_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("tools", ToolNode(self.tools))
        workflow.add_node("finalize", self._finalize_node)
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges("agent", self._route_after_agent, {"tools": "tools", "finalize": "finalize"})
        workflow.add_edge("tools", "agent")
        workflow.add_edge("finalize", END)
        return workflow.compile()

    def run(self, question: str, top_k: int = None) -> Dict[str, Any]:
        requested_top_k = top_k if top_k is not None else settings.DEFAULT_TOP_K
        initial_state: AgentState = {
            "question": question,
            "top_k": requested_top_k,
            "messages": [HumanMessage(content=question)],
            "retrieved_chunks": [],
            "retrieval_meta": {"candidate_count": 0, "final_count": 0, "retrieval_queries": [question]},
            "answer": "",
            "iterations": 0,
            "status": "running",
            "tool_history": [],
        }
        result = self.graph.invoke(initial_state)
        chunks = self._dedupe_chunks(result.get("retrieved_chunks", []))
        retrieval_meta = result.get("retrieval_meta", {"candidate_count": len(chunks), "final_count": len(chunks), "retrieval_queries": [question]})
        answer = result.get("answer") or self.rag_service.generate_answer(question, [chunk.get("text", "") for chunk in chunks if chunk.get("text")])
        return {
            "question": question,
            "answer": answer,
            "retrieval": {
                "candidate_count": int(retrieval_meta.get("candidate_count", len(chunks))),
                "final_count": int(retrieval_meta.get("final_count", len(chunks))),
                "retrieval_queries": [str(item) for item in retrieval_meta.get("retrieval_queries", [question])],
            },
            "retrieved_chunks": [
                {
                    "chunk_id": chunk.get("chunk_id", ""),
                    "filename": chunk.get("filename", "unknown.pdf"),
                    "page": chunk.get("page"),
                    "dense_distance": float(chunk.get("dense_distance", 0.0)),
                    "dense_similarity": float(chunk.get("dense_similarity", 1.0 - float(chunk.get("dense_distance", 0.0)))),
                    "reranker_score": float(chunk["reranker_score"]) if chunk.get("reranker_score") is not None else None,
                    "text": chunk.get("text", ""),
                }
                for chunk in chunks
            ],
        }
