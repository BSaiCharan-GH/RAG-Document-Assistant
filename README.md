# Intelligent RAG Document Assistant

A local document question-answering service built with FastAPI, ChromaDB, sentence-transformers, PyPDF, and Gemini. Upload a PDF, index it into a persistent vector store, retrieve the most relevant chunks for a question, and generate an answer using only the retrieved document context.

## Architecture

```text
PDF upload
   ↓
PyPDF text extraction
   ↓
Page-aware chunking
   ↓
all-MiniLM-L6-v2 embeddings (batch)
   ↓
ChromaDB (persistent local vector store)
   ↓
Semantic retrieval
   ↓
Gemini answer generation
   ↓
FastAPI REST API
   ↓
HTML / CSS / JavaScript UI
```

## Features

- PDF upload with file-size and extension validation.
- SHA-256 duplicate-document detection.
- Page-aware text extraction and chunking.
- Batch embedding during ingestion for better performance.
- Persistent ChromaDB storage.
- Configurable retrieval count, chunk size, and overlap.
- Gemini answers constrained to retrieved document context.
- Indexed-document listing and deletion.
- Source chunks displayed with filename, page, and retrieval distance.
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
│   ├── rag.py
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
- Internet access for the first download of the embedding model and Gemini requests

## Windows setup

Open PowerShell in the project root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Configure Gemini

Create `.env` in the project root by copying `.env.example`:

```powershell
Copy-Item .env.example .env
```

Then edit `.env` and set:

```env
GEMINI_API_KEY=your_new_api_key_here
GEMINI_MODEL=gemini-3.6-flash
```

If your Gemini account exposes a different model, set `GEMINI_MODEL` to that model name instead. Never put the API key in frontend files or commit `.env`.

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

### `GET /documents`

Returns all indexed documents.

### `DELETE /documents/{document_id}`

Deletes the selected document from ChromaDB and local upload storage.

## RAG behavior

The service first retrieves the requested number of semantically similar chunks. Gemini receives the question and those retrieved chunks as context. The prompt instructs Gemini not to rely on information outside that context and to state that the answer is not available in the provided document when the retrieved context is insufficient.

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

### Embedding model download

The first startup downloads `all-MiniLM-L6-v2`. Subsequent starts reuse the local model cache.

### No documents available

Upload a PDF through the UI or `POST /upload` before calling `POST /query`.

### Port already in use

Start on another port:

```powershell
uvicorn backend.main:app --reload --port 8001
```

## Development notes

The application initializes the embedding model, ChromaDB collection, and Gemini client once when the FastAPI process starts. Uploaded chunks are embedded in a batch rather than making one embedding call per chunk.
