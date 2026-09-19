# REST API based RAG Document Assistant

A local document question-answering service built with FastAPI, ChromaDB, sentence-transformers, PyPDF, and Gemini. The retrieval flow now uses a multi-stage pipeline: query processing, dense retrieval, candidate expansion, reranking, post-retrieval filtering, and final context selection before Gemini generates the final answer.

## Architecture

```text
User Query
    ↓
Pre-Retrieval
    ↓
Query Expansion
    ↓
Dense Retrieval
    ↓
Candidate Pool
    ↓
Cross-Encoder Re-Ranking
    ↓
Post-Retrieval Filtering
    ↓
Final Context
    ↓
Gemini
    ↓
Answer + Ranked Sources
```

## Advanced retrieval pipeline

### 1. Pre-retrieval

The system preserves the original user question and creates retrieval variants only when they add usefulness. This stage normalizes whitespace, removes redundant formatting, validates the input, and prevents empty queries. The original question is always preserved for the final Gemini answer, while the expanded variants are used only for retrieval.

### 2. Query expansion

The lightweight expansion logic keeps simple factual questions short and avoids unnecessary noise. For longer or more descriptive queries, it creates a few compact keyword-based variants for the dense search stage. All of this happens before ChromaDB retrieval.

### 3. Dense retrieval

The embedding model remains `all-MiniLM-L6-v2`, and ChromaDB remains the persistent vector database. Dense search uses cosine distance from ChromaDB, and the app converts it to a similarity value using:

```text
dense_similarity = 1 - dense_distance
```

A larger candidate pool is retrieved first, then the results are merged, deduplicated, and reranked.

### 4. Candidate retrieval and reranking

The system retrieves a configurable candidate pool, such as 20 chunks, for each query variant. Results are merged by `chunk_id` and the best dense match for each chunk is kept. After dense retrieval, the candidate list is sent to a local cross-encoder using `cross-encoder/ms-marco-MiniLM-L-6-v2`. The cross-encoder receives the original user query and the candidate text, and returns a relevance score.

### 5. Post-retrieval processing

The post-retrieval stage removes duplicate chunk IDs, filters near-duplicate content, removes clearly irrelevant chunks using the configured reranker threshold, preserves useful page diversity when possible, and selects the final top-k results while staying within `MAX_CONTEXT_CHARS`.

### 6. Final context selection

Only the final, reranked, filtered top-k chunks are passed to Gemini. The candidate pool is never sent to the model, and irrelevant chunks are excluded before generation.

### 7. Gemini generation

Gemini is used only as the final answer generator. It receives the original user question and the final context. It must answer using only the retrieved context and respond with:

```text
The answer is not available in the provided document.
```

when the context is insufficient.

## Features

- PDF upload with file-size and extension validation.
- SHA-256 duplicate-document detection.
- Page-aware text extraction and chunking.
- Batch embedding during ingestion for better performance.
- Persistent ChromaDB storage.
- Multi-stage retrieval pipeline with query normalization, expansion, dense search, reranking, and filtering.
- Configurable candidate and final top-k values, reranker threshold, and max context size.
- Gemini answers constrained to the final retrieved context only.
- Indexed-document listing and deletion.
- Source chunks displayed with filename, page, dense similarity, reranker score, and retrieved text.
- FastAPI Swagger documentation.
- Frontend rendering that treats document/model content as text instead of injecting it as HTML.

## Project structure

```text
RAG-Document-Assistant/
├── backend/
│   ├── __init__.py
│   ├── config.py
│   ├── embeddings.py
│   ├── main.py
│   ├── models.py
│   ├── pdf_processor.py
│   ├── query_processor.py
│   ├── rag.py
│   ├── retrieval.py
│   └── vector_store.py
├── frontend/
│   ├── app.js
│   ├── index.html
│   └── style.css
├── data/
│   ├── chroma/
│   └── uploads/
├── .env
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.10 or newer
- VS Code or another code editor
- A Gemini API key
- Internet access for the first download of the embedding model, reranker, and Gemini requests

## Windows setup

Open PowerShell in the project root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Configure environment variables

Create `.env` in the project root by copying `.env.example`:

```powershell
Copy-Item .env.example .env
```

Then set the values you need, including:

```env
GEMINI_API_KEY=your_new_api_key_here
GEMINI_MODEL=gemini-3.6-flash
EMBEDDING_MODEL=all-MiniLM-L6-v2
CANDIDATE_TOP_K=20
FINAL_TOP_K=4
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RERANKER_THRESHOLD=0.0
MAX_CONTEXT_CHARS=12000
```

Never put API keys in frontend files or commit `.env`.

## Start the service

From the project root, with the virtual environment active:

```powershell
python -m uvicorn backend.main:app --reload
```

Open:

- UI: `http://127.0.0.1:8000/`
- Swagger: `http://127.0.0.1:8000/docs`
- Health: `http://127.0.0.1:8000/health`

## REST API

### `GET /health`

Checks that the service is running.

### `POST /upload`

Accepts a PDF as multipart form data, extracts its text, creates chunks, embeds them in a batch, and stores them in ChromaDB.

Example PowerShell request:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/upload" -F "file=@example.pdf"
```

### `POST /query`

Request body:

```json
{
  "query": "Explain the Transformer architecture.",
  "top_k": 4
}
```

The API response includes:

```json
{
  "question": "Explain the Transformer architecture.",
  "answer": "...",
  "retrieval": {
    "candidate_count": 20,
    "final_count": 4,
    "retrieval_queries": ["Explain the Transformer architecture."]
  },
  "retrieved_chunks": [
    {
      "chunk_id": "...",
      "filename": "example.pdf",
      "page": 10,
      "dense_distance": 0.23,
      "dense_similarity": 0.77,
      "reranker_score": 4.12,
      "text": "..."
    }
  ]
}
```

### `GET /documents`

Returns all indexed documents.

### `DELETE /documents/{document_id}`

Deletes the selected document from ChromaDB and local upload storage.

## Data and secrets

The following are intentionally ignored by Git:

```text
.env
.venv/
data/chroma/
data/uploads/
```

Do not commit API keys, uploaded PDFs, or the local ChromaDB database.

## Troubleshooting

### Gemini model error

Check `GEMINI_API_KEY` and `GEMINI_MODEL` in `.env`. The model must be available to the Gemini API key you are using.

### Embedding or reranker model download

The first startup downloads `all-MiniLM-L6-v2` and the cross-encoder reranker. Subsequent starts reuse the local model cache.

### No documents available

Upload a PDF through the UI or `POST /upload` before calling `POST /query`.

### Port already in use

Start on another port:

```powershell
uvicorn backend.main:app --reload --port 8001
```

## Development notes

The application initializes the embedding model, reranker, ChromaDB collection, and Gemini client once when the FastAPI process starts. Uploaded chunks are embedded in a batch rather than making one embedding call per chunk, and retrieval now uses a modular stage-based pipeline for easier improvement later.
