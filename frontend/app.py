import os
from typing import Any, Dict, List

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Intelligent RAG Document Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("Intelligent RAG Document Assistant")
st.caption("Ask questions about your documents using Retrieval-Augmented Generation.")


@st.cache_data(show_spinner=False)
def check_backend_health(api_base_url: str) -> tuple[bool, str]:
    try:
        response = requests.get(f"{api_base_url}/health", timeout=10)
        if response.ok:
            data = response.json()
            return True, data.get("status", "ok")
        return False, f"Health check failed ({response.status_code})"
    except requests.RequestException as exc:
        return False, f"Backend unavailable: {exc}"


@st.cache_data(show_spinner=False)
def get_documents(api_base_url: str) -> List[Dict[str, Any]]:
    try:
        response = requests.get(f"{api_base_url}/documents", timeout=15)
        response.raise_for_status()
        payload = response.json()
        return payload.get("documents", [])
    except requests.RequestException:
        return []


@st.cache_data(show_spinner=False)
def delete_document(api_base_url: str, document_id: str) -> bool:
    try:
        response = requests.delete(f"{api_base_url}/documents/{document_id}", timeout=20)
        return response.ok
    except requests.RequestException:
        return False


@st.cache_data(show_spinner=False)
def upload_document(api_base_url: str, file_name: str, file_bytes: bytes) -> Dict[str, Any]:
    try:
        response = requests.post(
            f"{api_base_url}/upload",
            files={"file": (file_name, file_bytes, "application/pdf")},
            timeout=120,
        )
        payload = response.json() if response.content else {}
        if response.ok:
            return payload
        raise ValueError(payload.get("detail", "Upload failed."))
    except (requests.RequestException, ValueError) as exc:
        raise RuntimeError(str(exc)) from exc


@st.cache_data(show_spinner=False)
def query_backend(api_base_url: str, question: str, top_k: int = 4) -> Dict[str, Any]:
    try:
        response = requests.post(
            f"{api_base_url}/query",
            json={"query": question, "top_k": top_k},
            timeout=120,
        )
        payload = response.json() if response.content else {}
        if response.ok:
            return payload
        detail = payload.get("detail", "Query failed.")
        if isinstance(detail, list):
            detail = detail[0].get("msg", str(detail)) if detail else "Query failed."
        raise ValueError(str(detail))
    except (requests.RequestException, ValueError) as exc:
        raise RuntimeError(str(exc)) from exc


with st.sidebar:
    st.subheader("Backend")
    backend_url = st.text_input("API Base URL", value=API_BASE_URL, key="backend_url")
    health_ok, health_message = check_backend_health(backend_url)
    st.write(f"Backend: {'Connected' if health_ok else 'Unavailable'}")
    if health_ok:
        st.success(f"Status: {health_message}")
    else:
        st.error(health_message)

    st.markdown("---")
    st.subheader("Indexed Documents")
    docs = get_documents(backend_url)
    if docs:
        st.metric("Documents", len(docs))
        for doc in docs:
            st.caption(f"{doc.get('filename', 'unknown.pdf')} • {doc.get('chunks_count', 0)} chunks")
    else:
        st.info("No indexed documents yet.")


with st.container():
    col_upload, col_documents = st.columns([1.2, 1])

    with col_upload:
        st.subheader("Upload Document")
        uploaded_file = st.file_uploader(
            "Choose a PDF",
            type=["pdf"],
            accept_multiple_files=False,
            help="Upload a PDF document to be indexed and made searchable.",
        )

        if uploaded_file is not None:
            st.write(f"Filename: {uploaded_file.name}")
            st.write(f"File size: {uploaded_file.size} bytes")

        upload_button = st.button("Upload PDF", type="primary", disabled=uploaded_file is None or not health_ok)

        if upload_button and uploaded_file is not None:
            with st.spinner("Indexing document... This may take a few minutes for large documents."):
                try:
                    result = upload_document(backend_url, uploaded_file.name, uploaded_file.getvalue())
                    st.success("Document indexed successfully")
                    st.write(f"Filename: {result.get('filename', uploaded_file.name)}")
                    st.write(f"Pages: {result.get('pages', 0)}")
                    st.write(f"Chunks: {result.get('chunks_created', 0)}")
                    st.write(f"Status: {result.get('status', 'Indexed')}")
                    st.session_state["uploaded_document_name"] = result.get("filename", uploaded_file.name)
                    st.cache_data.clear()
                    st.rerun()
                except Exception as exc:
                    st.error(f"Upload failed: {exc}")

    with col_documents:
        st.subheader("Indexed Documents")
        col_refresh, col_delete = st.columns([1, 1])
        with col_refresh:
            if st.button("Refresh"):
                st.cache_data.clear()
                st.rerun()

        docs = get_documents(backend_url)
        if not docs:
            st.info("No indexed documents available.")
        else:
            for doc in docs:
                with st.container():
                    st.markdown(f"**{doc.get('filename', 'unknown.pdf')}**")
                    st.write(f"Pages: {len(doc.get('pages', []))}")
                    st.write(f"Chunks: {doc.get('chunks_count', 0)}")
                    st.write(f"Status: {doc.get('status', 'indexed')}")
                    if st.button(f"Delete {doc.get('filename', 'document')}", key=f"delete_{doc.get('document_id', '')}"):
                        if delete_document(backend_url, doc.get("document_id", "")):
                            st.success("Document deleted.")
                            st.cache_data.clear()
                            st.rerun()
                        else:
                            st.error("Failed to delete the document.")
                    st.markdown("---")

st.markdown("---")

st.subheader("Ask a Question")
question = st.text_area(
    "",
    placeholder="Ask something about the uploaded documents...",
    height=150,
)
ask_button = st.button("Ask", type="primary", disabled=not question.strip() or not health_ok)

if ask_button and question.strip():
    with st.spinner("Searching the documents and generating an answer..."):
        try:
            payload = query_backend(backend_url, question.strip(), top_k=4)
            st.session_state["last_answer"] = payload.get("answer", "")
            st.session_state["last_query"] = payload.get("question", question.strip())
            st.session_state["last_retrieval"] = payload.get("retrieval", {})
            st.session_state["last_sources"] = payload.get("retrieved_chunks", [])
            st.session_state["last_source"] = payload.get("source", {})
            st.session_state["last_web_sources"] = payload.get("web_sources", [])
        except Exception as exc:
            st.error(f"Query failed: {exc}")
            st.session_state["last_answer"] = ""
            st.session_state["last_query"] = question.strip()
            st.session_state["last_retrieval"] = {}
            st.session_state["last_sources"] = []
            st.session_state["last_source"] = {}
            st.session_state["last_web_sources"] = []

if "last_answer" in st.session_state and st.session_state["last_answer"]:
    source = st.session_state.get("last_source", {})
    source_mode = str(source.get("mode", "document")).lower()
    source_labels = {
        "document": "DOCUMENTS",
        "web": "WEB SEARCH",
        "hybrid": "DOCUMENTS + WEB",
    }
    st.info(f"Source strategy: {source_labels.get(source_mode, source_mode.upper())}\n\n{source.get('explanation', '')}")
    st.subheader("Answer")
    st.markdown(st.session_state["last_answer"])

    retrieval = st.session_state.get("last_retrieval", {})
    if retrieval:
        st.subheader("Retrieval Information")
        st.write(f"Candidate chunks: {retrieval.get('candidate_count', 0)}")
        st.write(f"Final chunks: {retrieval.get('final_count', 0)}")
        queries = retrieval.get("retrieval_queries", [])
        if queries:
            st.write("Retrieval queries:")
            for item in queries:
                st.markdown(f"- {item}")

    sources = st.session_state.get("last_sources", [])
    if sources:
        st.subheader("Ranked Sources")
        for index, chunk in enumerate(sources, start=1):
            filename = chunk.get("filename", "unknown.pdf")
            page = chunk.get("page")
            dense_similarity = chunk.get("dense_similarity")
            dense_distance = chunk.get("dense_distance")
            reranker_score = chunk.get("reranker_score")

            with st.expander(f"Source {index}: {filename} • Page {page}"):
                st.write(f"Document: {filename}")
                st.write(f"Page: {page}")
                st.write(f"Dense distance: {dense_distance}")
                st.write(f"Dense similarity: {dense_similarity}")
                st.write(f"Reranker score: {reranker_score}")
                st.write("Dense similarity: Similarity from the dense embedding retrieval stage.")
                st.write("Dense distance: Distance returned by ChromaDB.")
                st.write("Reranker score: Cross-encoder ranking score.")
                st.write("Retrieved text:")
                st.write(chunk.get("text", ""))
    else:
        st.info("No sources were returned for this query.")

    web_sources = st.session_state.get("last_web_sources", [])
    if web_sources:
        st.subheader("Web Sources")
        for index, source in enumerate(web_sources, start=1):
            title = source.get("title") or source.get("domain") or "Web source"
            domain = source.get("domain", "")
            snippet = source.get("snippet", "")
            url = source.get("url", "")
            st.markdown(f"**{index}. {title}**  \n{domain}  \n{snippet}")
            if url:
                st.link_button("Open source", url)

if not health_ok:
    st.warning("The backend is currently unavailable. Start the FastAPI service and confirm the API URL.")
