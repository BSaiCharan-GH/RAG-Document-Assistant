# PDF RAG Assistant

- PDF question-answering service built with FastAPI, ChromaDB, sentence-transformers, PyPDF, and Gemini. The application ingests uploaded PDFs, stores text chunks and vector embeddings locally, retrieves the most relevant chunks for a query, and uses Gemini to answer questions strictly from the retrieved context.

## Overview

This project follows a complete retrieval-augmented generation pipeline:

PDF → text extraction → chunking → embedding → ChromaDB → similarity retrieval → Gemini → FastAPI API → frontend UI

## Architecture

- Backend: FastAPI REST API in the `backend` package
- Frontend: plain HTML, CSS, and JavaScript in the `frontend` folder
- Vector database: ChromaDB persisted locally under `data/chroma`
- Upload storage: local file storage under `data/uploads`
- Embedding model: `all-MiniLM-L6-v2` via `sentence-transformers`
- LLM: Gemini via the Google GenAI SDK

## Folder structure

```text
pdf-rag-service/
├── backend/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── models.py
│   ├── rag.py
│   ├── embeddings.py
│   ├── vector_store.py
│   └── pdf_processor.py
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
├── data/
│   ├── uploads/
│   └── chroma/
├── .env
├── .env.example
├── .gitignore
├── requirements.txt
├── README.md
└── .venv/
```

## Requirements

- Python 3.10+
- Windows 10 or 11
- VS Code
- A Gemini API key

## Installation

Open PowerShell in the project directory and create a virtual environment:

```powershell
cd path\to\project
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Creating .env

Create a `.env` file in the project root based on `.env.example`:

```env
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-2.0-flash
CHROMA_PATH=./data/chroma
UPLOAD_PATH=./data/uploads
EMBEDDING_MODEL=all-MiniLM-L6-v2
DEFAULT_TOP_K=4
CHUNK_SIZE=800
CHUNK_OVERLAP=150
MAX_UPLOAD_SIZE_MB=20
MAX_QUERY_LENGTH=500
```

## Adding the Gemini API key

Set your actual Gemini API key in `.env`:

```env
GEMINI_API_KEY=your_real_api_key_here
```

Do not commit `.env` to source control. It is ignored by `.gitignore`.

## Starting the service

From the project root:

```powershell
.venv\Scripts\Activate.ps1
uvicorn backend.main:app --reload
```

The API is then available at:

- UI: http://127.0.0.1:8000/
- Swagger: http://127.0.0.1:8000/docs
- Health: http://127.0.0.1:8000/health

## Using the upload feature

1. Open the UI at http://127.0.0.1:8000/
2. Drag and drop a PDF or click Browse.
3. Click Upload PDF.
4. The document is validated, saved locally, chunked, embedded, and indexed.

## Asking questions

1. Type a question in the input box.
2. Click Ask.
3. The backend retrieves relevant chunks from ChromaDB and asks Gemini to answer using only that context.

## REST API endpoints

- `GET /health`
- `POST /upload`
- `POST /query`
- `GET /documents`
- `DELETE /documents/{document_id}`
- `GET /docs` (Swagger UI)

## Example API requests

### Health check

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health"
```

### Upload a PDF

```powershell
curl -X POST "http://127.0.0.1:8000/upload" -F "file=@example.pdf"
```

### Ask a question

```powershell
curl -X POST "http://127.0.0.1:8000/query" -H "Content-Type: application/json" -d '{"query":"Explain the architecture in this document","top_k":4}'
```

### List documents

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/documents"
```

### Delete a document

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/documents/<document_id>" -Method Delete
```

## Swagger documentation

Swagger UI is available automatically via FastAPI at:

http://127.0.0.1:8000/docs

## Troubleshooting

- If the model fails to load, ensure `sentence-transformers` has downloaded the `all-MiniLM-L6-v2` model.
- If Gemini returns errors, verify the API key in `.env` and the model name.
- If uploads fail, confirm the file is a valid PDF and under the upload size limit.
- If the UI does not load, run the app from the project root with `uvicorn backend.main:app --reload`.
- If ChromaDB is empty, upload a PDF first.

## Duplicate document behavior

If the same PDF is uploaded again, the service detects it using a SHA-256 hash and reports a duplicate instead of indexing it twice. This avoids unnecessary duplicate embeddings and keeps the local index clean.

## Notes on retrieval and answer quality

The generated answer is only based on the retrieved context. If the retrieved chunks do not contain sufficient information, the app explicitly states that the answer is not available in the provided document.
