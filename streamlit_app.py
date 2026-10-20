"""Single-service Streamlit deployment for the existing Agentic RAG Document Assistant.

The deployed UI calls the repository's existing backend services in-process, so it
can run on Streamlit Community Cloud without a separate FastAPI host. Each browser
session gets its own Chroma directory and uploaded-file directory.
"""
from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path
from typing import Any

import streamlit as st

st.set_page_config(
    page_title="Agentic RAG Document Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _load_streamlit_secrets_into_environment() -> None:
    """Expose Streamlit Cloud secrets to the project's existing settings loader."""
    for name in (
        "GEMINI_API_KEY",
        "TAVILY_API_KEY",
        "GEMINI_MODEL",
        "AGENT_MODEL",
        "EMBEDDING_MODEL",
        "RERANKER_MODEL",
    ):
        try:
            value = st.secrets.get(name, "")
        except Exception:
            value = ""
        if value and str(value).strip():
            os.environ[name] = str(value).strip()


_load_streamlit_secrets_into_environment()

# Import project services only after reading Streamlit secrets, because settings
# are initialized from environment variables when backend.config is imported.
from sentence_transformers import CrossEncoder  # noqa: E402

from backend.agent import AgenticRAGService  # noqa: E402
from backend.config import settings  # noqa: E402
from backend.embeddings import EmbeddingService  # noqa: E402
from backend.pdf_processor import (  # noqa: E402
    chunk_text,
    compute_file_hash,
    extract_pdf_text,
    sanitize_filename,
)
from backend.rag import RAGService  # noqa: E402
from backend.retrieval import RetrievalService  # noqa: E402
from backend.query_processor import QueryProcessor  # noqa: E402
from backend.vector_store import VectorStore  # noqa: E402


# A shared, read-only copy of each ML model prevents every browser session from
# downloading and keeping its own copy in memory. Per-session indices remain
# separate because the VectorStore is created separately for each Streamlit session.
@st.cache_resource(show_spinner="Loading the sentence-transformer embedding model…")
def _get_embedding_service() -> EmbeddingService:
    return EmbeddingService()


@st.cache_resource(show_spinner="Loading the cross-encoder reranker…")
def _get_reranker() -> CrossEncoder:
    return CrossEncoder(settings.RERANKER_MODEL, max_length=512)


class _SessionRetrievalService(RetrievalService):
    """Reuse the project's retrieval methods with a process-cached reranker."""

    def __init__(self, vector_store: VectorStore, embedding_service: EmbeddingService, reranker: Any) -> None:
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.query_processor = QueryProcessor(max_length=settings.MAX_QUERY_LENGTH)
        self.reranker = reranker


def _workspace_root() -> Path:
    session_id = st.session_state.get("_rag_session_id")
    if not session_id:
        session_id = uuid.uuid4().hex
        st.session_state["_rag_session_id"] = session_id
    root = Path(tempfile.gettempdir()) / "rag_document_assistant_sessions" / session_id
    (root / "uploads").mkdir(parents=True, exist_ok=True)
    (root / "chroma").mkdir(parents=True, exist_ok=True)
    return root


def _get_session_services() -> tuple[VectorStore, EmbeddingService, _SessionRetrievalService, RAGService, AgenticRAGService, Path]:
    if "_rag_services" not in st.session_state:
        root = _workspace_root()
        store = VectorStore(persist_dir=root / "chroma")
        embedder = _get_embedding_service()
        retrieval = _SessionRetrievalService(store, embedder, _get_reranker())
        rag = RAGService(vector_store=store, embedding_service=embedder, retrieval_service=retrieval)
        agent = AgenticRAGService(
            vector_store=store,
            embedding_service=embedder,
            retrieval_service=retrieval,
            rag_service=rag,
        )
        st.session_state["_rag_services"] = (store, embedder, retrieval, rag, agent, root)
    return st.session_state["_rag_services"]


MAX_DOCUMENTS_PER_SESSION = 5
MAX_TOTAL_UPLOAD_MB_PER_SESSION = 60
MAX_QUERIES_PER_SESSION = 20


def _index_pdf(uploaded_file: Any, store: VectorStore, embedder: EmbeddingService, root: Path) -> dict[str, Any]:
    file_bytes = uploaded_file.getvalue()
    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if len(file_bytes) > max_bytes:
        raise ValueError(f"This PDF exceeds the {settings.MAX_UPLOAD_SIZE_MB} MB per-file upload limit.")
    if not file_bytes:
        raise ValueError("The selected PDF is empty.")

    existing_documents = store.list_documents()
    if len(existing_documents) >= MAX_DOCUMENTS_PER_SESSION:
        raise ValueError(f"This demo allows up to {MAX_DOCUMENTS_PER_SESSION} indexed PDFs per browser session.")
    upload_dir = root / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    existing_bytes = sum(path.stat().st_size for path in upload_dir.glob("*.pdf") if path.is_file())
    max_total_bytes = MAX_TOTAL_UPLOAD_MB_PER_SESSION * 1024 * 1024
    if existing_bytes + len(file_bytes) > max_total_bytes:
        raise ValueError(f"The total PDF upload limit for this session is {MAX_TOTAL_UPLOAD_MB_PER_SESSION} MB.")

    document_id = compute_file_hash(file_bytes)
    safe_name = sanitize_filename(uploaded_file.name)
    if store.has_document(document_id):
        raise ValueError("This exact PDF is already indexed in your current session.")

    saved_path = upload_dir / f"{document_id}.pdf"
    saved_path.write_bytes(file_bytes)

    try:
        extracted_text, page_count = extract_pdf_text(saved_path)
        if not extracted_text.strip():
            raise ValueError("No selectable text was found in this PDF. Scanned PDFs need OCR before upload.")
        chunks = chunk_text(extracted_text)
        texts = [chunk.text for chunk in chunks]
        embeddings = embedder.embed_texts(texts)
        metadatas = [
            {
                "document_id": document_id,
                "filename": safe_name,
                "page": int(chunk.page) if chunk.page is not None else 0,
                "chunk_id": f"{document_id}_{index}",
                "source": "uploaded_pdf",
            }
            for index, chunk in enumerate(chunks)
        ]
        count = store.add_document(document_id, safe_name, texts, metadatas, embeddings)
        if count == 0:
            raise ValueError("No searchable text chunks were created from this PDF.")
        return {
            "filename": safe_name,
            "document_id": document_id,
            "pages": page_count,
            "chunks_created": count,
        }
    except Exception:
        try:
            store.delete_document(document_id)
        except Exception:
            pass
        try:
            saved_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _delete_document(document_id: str, store: VectorStore, root: Path) -> bool:
    deleted = store.delete_document(document_id)
    file_path = root / "uploads" / f"{document_id}.pdf"
    try:
        file_path.unlink(missing_ok=True)
    except OSError:
        pass
    return deleted


def _render_answer(result: dict[str, Any]) -> None:
    source = result.get("source", {})
    mode = str(source.get("mode", "unknown")).upper()
    mode_label = {"DOCUMENT": "DOCUMENTS", "WEB": "WEB SEARCH", "HYBRID": "DOCUMENTS + WEB"}.get(mode, mode)

    st.subheader("Answer")
    st.markdown(result.get("answer") or "No answer was returned.")
    st.caption(f"Source strategy: {mode_label} · {source.get('explanation', '')}")

    retrieval = result.get("retrieval", {})
    agent = result.get("agent", {})
    c1, c2, c3 = st.columns(3)
    c1.metric("Candidate chunks", int(retrieval.get("candidate_count", 0)))
    c2.metric("Final evidence chunks", int(retrieval.get("final_count", 0)))
    c3.metric("Agent steps", int(agent.get("iterations", 0)))
    tools_used = agent.get("tools_used", [])
    st.caption("Tools used: " + (", ".join(tools_used) if tools_used else "none reported"))

    retrieved_chunks = result.get("retrieved_chunks", [])
    if retrieved_chunks:
        st.subheader("Document evidence")
        for index, chunk in enumerate(retrieved_chunks, start=1):
            title = f"{index}. {chunk.get('filename', 'Document')} · page {chunk.get('page', 'N/A')}"
            with st.expander(title, expanded=(index == 1)):
                s1, s2 = st.columns(2)
                s1.metric("Dense similarity", f"{float(chunk.get('dense_similarity', 0.0)):.4f}")
                reranker_score = chunk.get("reranker_score")
                s2.metric("Reranker score", f"{float(reranker_score):.4f}" if reranker_score is not None else "N/A")
                st.write(chunk.get("text", ""))

    web_sources = result.get("web_sources", [])
    if web_sources:
        st.subheader("Web sources")
        for index, source_item in enumerate(web_sources, start=1):
            title = str(source_item.get("title") or source_item.get("domain") or f"Web result {index}")
            st.write(f"**{title}**")
            snippet = source_item.get("snippet") or source_item.get("content")
            if snippet:
                st.write(snippet)
            url = str(source_item.get("url", ""))
            if url.startswith(("https://", "http://")):
                st.link_button("Open source", url)

    with st.expander("Agent and retrieval trace"):
        st.json({
            "source": source,
            "agent": agent,
            "retrieval": retrieval,
        })


# ------------------------------- Page UI ------------------------------------
st.markdown(
    """
    <style>
      .block-container {max-width: 1280px; padding-top: 2rem; padding-bottom: 3rem;}
      .hero {padding: 1.2rem 1.4rem; border: 1px solid rgba(120,160,210,.22);
             border-radius: 18px; background: linear-gradient(120deg, rgba(30,65,110,.30), rgba(12,24,42,.12));}
      .hero p {color: #9eb4cf; margin-bottom: 0;}
      div[data-testid="stMetric"] {border: 1px solid rgba(120,160,210,.20); padding: .8rem; border-radius: 12px;}
    </style>
    <div class="hero">
      <h1>📚 Agentic RAG Document Assistant</h1>
      <p>Document-grounded answers · adaptive source selection · semantic retrieval · cross-encoder reranking · evidence provenance</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")

if not settings.GEMINI_API_KEY:
    st.error("Gemini is not configured yet. Add GEMINI_API_KEY in Streamlit Community Cloud → App settings → Secrets, then restart the app.")
    st.markdown("Create a key in [Google AI Studio](https://aistudio.google.com/app/apikey). Do not paste your key into chat or commit it to GitHub.")
    st.code('GEMINI_API_KEY = "paste-your-key-in-Streamlit-Secrets"\nGEMINI_MODEL = "gemini-3.6-flash"\nAGENT_MODEL = "gemini-3.6-flash"', language="toml")
    st.stop()

if not settings.TAVILY_API_KEY:
    st.warning("TAVILY_API_KEY is not configured. Document-only question answering can work; live WEB and HYBRID retrieval requires a Tavily key.")

with st.spinner("Preparing the isolated workspace and retrieval models…"):
    vector_store, embedding_service, retrieval_service, rag_service, agent_service, workspace_root = _get_session_services()

with st.sidebar:
    st.markdown("### Workspace")
    documents = vector_store.list_documents()
    st.metric("Indexed documents", len(documents))
    st.metric("Indexed chunks", sum(int(doc.get("chunks_count", 0)) for doc in documents))
    query_count = int(st.session_state.get("_rag_query_count", 0))
    st.metric("Queries used", f"{query_count}/{MAX_QUERIES_PER_SESSION}")
    st.caption(f"Session limits: {MAX_DOCUMENTS_PER_SESSION} PDFs, {MAX_TOTAL_UPLOAD_MB_PER_SESSION} MB total, {MAX_QUERIES_PER_SESSION} queries. Each browser session has a separate index; temporary storage may reset when the app restarts.")
    st.divider()
    st.markdown("### Retrieval pipeline")
    st.caption("Dense retrieval → cross-encoder reranking → evidence evaluation → Gemini answer with source provenance")
    st.caption(f"Embedding: `{settings.EMBEDDING_MODEL}`")
    st.caption(f"Reranker: `{settings.RERANKER_MODEL}`")

upload_tab, query_tab, about_tab = st.tabs(["📄 Documents", "💬 Ask & Explore Evidence", "ℹ️ Methodology"])

with upload_tab:
    upload_col, info_col = st.columns([1.15, 1])
    with upload_col:
        st.subheader("Upload a PDF")
        uploaded_file = st.file_uploader(
            "Choose a PDF to index in this session",
            type=["pdf"],
            accept_multiple_files=False,
            help=f"Maximum size: {settings.MAX_UPLOAD_SIZE_MB} MB. Text-based PDFs are supported; scanned PDFs need OCR first.",
        )
        upload_clicked = st.button("Index PDF", type="primary", disabled=uploaded_file is None, use_container_width=True)
        if upload_clicked and uploaded_file is not None:
            try:
                with st.spinner("Extracting pages, creating embeddings and indexing chunks…"):
                    summary = _index_pdf(uploaded_file, vector_store, embedding_service, workspace_root)
                st.success(f"Indexed {summary['filename']}: {summary['pages']} pages and {summary['chunks_created']} chunks.")
                st.rerun()
            except Exception as exc:
                st.error(f"Could not index the PDF: {exc}")
    with info_col:
        st.subheader("What happens after upload")
        st.markdown(
            "1. Extract text and page numbers with **pypdf**.\n"
            "2. Split the text into overlapping chunks.\n"
            "3. Create local Sentence Transformer embeddings.\n"
            "4. Store vectors in this session's ChromaDB collection.\n"
            "5. Retrieve and rerank evidence when you ask a question."
        )
        st.info("Upload only documents you are permitted to process. Avoid confidential or sensitive files in a public demo.")

    st.divider()
    st.subheader("Documents in this session")
    documents = vector_store.list_documents()
    if not documents:
        st.info("No documents indexed yet. Upload a text-based PDF above.")
    else:
        for document in documents:
            with st.container(border=True):
                doc_col, action_col = st.columns([4, 1])
                with doc_col:
                    st.markdown(f"**{document.get('filename', 'unknown.pdf')}**")
                    pages = document.get("pages", [])
                    st.caption(f"{len(pages)} pages represented · {document.get('chunks_count', 0)} chunks · ID `{document.get('document_id', '')[:12]}…`")
                with action_col:
                    if st.button("Remove", key=f"remove_{document.get('document_id')}", use_container_width=True):
                        _delete_document(str(document.get("document_id", "")), vector_store, workspace_root)
                        st.success("Document removed from this session.")
                        st.rerun()

with query_tab:
    st.subheader("Ask your documents")
    st.caption("The existing LangGraph workflow selects DOCUMENT, WEB or HYBRID retrieval according to the question and indexed-document availability.")
    question = st.text_area(
        "Question",
        placeholder="Example: According to the uploaded document, explain the key architecture and compare it with current practice.",
        height=110,
        max_chars=settings.MAX_QUERY_LENGTH,
    )
    qcol1, qcol2 = st.columns([1, 3])
    with qcol1:
        top_k = st.slider("Evidence passages", min_value=1, max_value=8, value=settings.FINAL_TOP_K)
    with qcol2:
        st.caption("DOCUMENT: indexed PDFs · WEB: current external information · HYBRID: document evidence plus web context")
    ask_clicked = st.button("Run agentic retrieval", type="primary", disabled=not question.strip(), use_container_width=True)

    if ask_clicked and question.strip():
        query_count = int(st.session_state.get("_rag_query_count", 0))
        if query_count >= MAX_QUERIES_PER_SESSION:
            st.error(f"This demo is limited to {MAX_QUERIES_PER_SESSION} queries per browser session.")
        else:
            st.session_state["_rag_query_count"] = query_count + 1
            try:
                with st.spinner("Selecting sources, retrieving evidence, reranking and generating the answer…"):
                    result = agent_service.run(question.strip(), top_k=top_k)
                st.session_state["_rag_last_result"] = result
                st.session_state["_rag_last_question"] = question.strip()
            except Exception as exc:
                st.error(f"The query could not be completed: {exc}")

    previous_result = st.session_state.get("_rag_last_result")
    if previous_result:
        st.divider()
        st.caption(f"Last question: {st.session_state.get('_rag_last_question', '')}")
        _render_answer(previous_result)

with about_tab:
    st.subheader("System design")
    st.markdown(
        """
        **This deployment reuses the repository's existing backend modules.** The original FastAPI API remains available for local/server use; this Streamlit entrypoint calls the same processing and agent services in-process to avoid requiring a separately hosted API for the demo.

        | Stage | Existing component |
        |---|---|
        | PDF text extraction and chunking | `backend/pdf_processor.py` |
        | Local vector storage | ChromaDB via `backend/vector_store.py` |
        | Dense embeddings | Sentence Transformers |
        | Candidate reranking | Cross-encoder (`ms-marco-MiniLM-L-6-v2`) |
        | Source/tool orchestration | LangGraph (`backend/agent.py`) |
        | Answer generation | Google Gemini |
        | Optional web retrieval | Tavily |

        **Operational notes:** uploaded files and Chroma indices are scoped to a random browser-session workspace. The demo caps each session at five PDFs, 60 MB total and 20 questions. The Streamlit container uses temporary storage, so content may disappear after an app restart. The public app is a portfolio demo, not a hardened multi-user document-management service; provider API quotas/costs may still apply.
        """
    )
    st.markdown("**Example questions**")
    st.code(
        'According to the uploaded document, explain the main architecture.\n'
        'What is the latest update related to this topic?\n'
        'Compare the uploaded document with current information from the web.',
        language="text",
    )
