# Agentic RAG Document Assistant

An intelligent document question-answering system that combines Retrieval-Augmented Generation (RAG), agentic decision-making, local document retrieval, web search, and evidence evaluation.

The system does not blindly search documents or the web for every question. A LangGraph-based agent analyzes each query and decides whether the answer should come from:

- **DOCUMENT** — uploaded documents only
- **WEB** — current information from the web using Tavily
- **HYBRID** — uploaded documents combined with web information

It then evaluates the collected evidence, performs additional retrieval when required, and generates a grounded answer with source information.

---

## Features

### Agentic Source Decision Making

The system dynamically determines the most appropriate information source for each query.

```text
User Query
    ↓
Question Analysis
    ↓
Source Decision
    ├── DOCUMENT
    ├── WEB
    └── HYBRID
    ↓
Evidence Retrieval
    ↓
Evidence Evaluation
    ↓
Sufficient?
   ├── Yes → Generate Answer
   └── No  → Refine / Retrieve Again
```

The decision considers factors such as:

- Whether the question refers to uploaded documents
- Whether current or external information is required
- Availability of relevant document evidence
- Whether web information is necessary
- Whether the retrieved evidence is sufficient

---

## Document Question Answering

Users can upload PDF documents and ask questions about their contents.

The document pipeline performs:

```text
PDF Upload
    ↓
Text Extraction
    ↓
Text Cleaning
    ↓
Chunking
    ↓
Embedding Generation
    ↓
ChromaDB
    ↓
Semantic Retrieval
    ↓
Reranking
    ↓
Relevant Context
```

The system preserves document metadata such as:

- Filename
- Page number
- Chunk identifier
- Retrieval score
- Reranker score

---

## Web Search with Tavily

For questions requiring current or external information, the agent can use Tavily web search.

Example:

```text
"What is the latest version of Python?"
        ↓
      WEB
        ↓
  Tavily Search
        ↓
   Web Sources
        ↓
Evidence Evaluation
        ↓
      Answer
```

The web search component provides:

- Search results
- Source titles
- URLs
- Domains
- Relevant snippets
- Current web information

Web sources are displayed separately from uploaded-document sources so users can understand where information originated.

---

## Hybrid Retrieval

Some questions require both the user's documents and external information.

Example:

> According to my uploaded document, how does this technology work today?

The agent can determine that both sources are necessary.

```text
                    User Query
                       ↓
                 Source Decision
                       ↓
                    HYBRID
                   /       \
                  ↓         ↓
             ChromaDB     Tavily
             Documents      Web
                  \         /
                   ↓       ↓
                 Evidence
                    ↓
              Evidence Check
                    ↓
              Grounded Answer
```

The final response can distinguish between information obtained from uploaded documents and information obtained from the web.

---

# Agentic Workflow

The application uses a graph-based workflow built with LangGraph.

```text
START
  ↓
Analyze Question
  ↓
Decide Source
  ↓
Execute Retrieval
  ├───────────────┐
  ↓               ↓
Documents        Tavily
  ↓               ↓
  └───────┬───────┘
          ↓
Evaluate Evidence
          ↓
   Evidence Sufficient?
       /          \
     Yes           No
      ↓             ↓
Generate Answer   Refine Query
      ↑             ↓
      └──── Retrieve Again
          ↓
         END
```

The agent maintains state throughout the workflow, including:

- User question
- Retrieval queries
- Retrieved document chunks
- Web search results
- Selected source
- Tools used
- Number of iterations
- Evidence sufficiency
- Final answer
- Source information

---

# Source Selection

The agent supports three retrieval modes.

| Mode | Purpose | Example |
|---|---|---|
| `DOCUMENT` | Answer using uploaded documents | "According to the uploaded textbook, explain attention." |
| `WEB` | Retrieve current or external information | "What is the latest Python release?" |
| `HYBRID` | Combine document and web evidence | "Compare the technology in my document with its current implementation." |

The selected source mode is exposed to the user through the interface.

---

# Advanced Document Retrieval

The document retrieval system uses a multi-stage retrieval pipeline rather than directly using the first few vector search results.

```text
User Query
    ↓
Query Processing
    ↓
Dense Retrieval
    ↓
Candidate Pool
    ↓
Cross-Encoder Reranking
    ↓
Duplicate Removal
    ↓
Relevance Filtering
    ↓
Context Optimization
    ↓
Final Context
```

### Dense Retrieval

The system uses Sentence Transformers to perform semantic similarity search.

Default embedding model:

```text
sentence-transformers/all-MiniLM-L6-v2
```

### Candidate Retrieval

A larger candidate pool is retrieved initially.

```text
Candidate Top K = 20
```

### Cross-Encoder Reranking

Candidates are reranked using:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

This allows the system to evaluate the relationship between the query and retrieved passages more precisely than vector similarity alone.

### Final Context

Only the most relevant results are passed into the answer-generation stage.

```text
Final Top K = 4
```

---

# Evidence Evaluation

The agent evaluates whether the retrieved information is sufficient before generating the final response.

If the evidence is insufficient, the agent can:

- Refine the retrieval query
- Search again
- Expand the retrieval process
- Use another information source when appropriate
- Perform another evidence evaluation cycle

This prevents the system from treating the first retrieval result as automatically sufficient.

---

# Query Expansion

When the original query does not retrieve enough useful information, the agent can generate an improved retrieval query.

Example:

```text
Original:
"Explain the architecture."

Expanded:
"Transformer architecture components attention feed-forward
network layer normalization positional encoding"
```

The expanded query is then sent through the retrieval pipeline.

---

# Source Provenance

The system keeps track of the sources used to generate an answer.

### Document Sources

Document results include information such as:

```text
Filename
Page
Chunk ID
Dense similarity
Reranker score
Relevant text
```

### Web Sources

Web results include:

```text
Title
URL
Domain
Snippet
```

This allows users to inspect the underlying information used by the system.

---

# Technology Stack

## Backend

- Python
- FastAPI
- Pydantic
- LangChain
- LangGraph
- Google Gemini
- ChromaDB
- Sentence Transformers
- Cross-Encoder

## Web Search

- Tavily

## Frontend

- Streamlit

## Document Processing

- PyPDF

## Machine Learning

- Sentence Transformers
- Dense embeddings
- Cross-encoder reranking

---

# Architecture

```text
┌─────────────────────────────┐
│       Streamlit UI          │
│                             │
│  Upload PDF / Ask Question  │
└──────────────┬──────────────┘
               │
               ↓
┌─────────────────────────────┐
│          FastAPI            │
│                             │
│  API / Upload / Query       │
└──────────────┬──────────────┘
               │
               ↓
┌─────────────────────────────┐
│       LangGraph Agent       │
│                             │
│ Analyze → Decide → Retrieve │
│              ↓              │
│       Evaluate Evidence     │
└───────┬───────────┬─────────┘
        │           │
        ↓           ↓
┌────────────┐ ┌──────────────┐
│ ChromaDB   │ │    Tavily    │
│            │ │              │
│ Documents  │ │ Web Search   │
└─────┬──────┘ └──────┬───────┘
      │               │
      └───────┬───────┘
              ↓
      ┌───────────────┐
      │    Evidence   │
      │    Evaluation │
      └───────┬───────┘
              ↓
      ┌───────────────┐
      │ Answer        │
      │ Generation    │
      └───────────────┘
```

---

# Project Structure

```text
RAG-Document-Assistant/
│
├── backend/
│   ├── main.py
│   ├── agent.py
│   ├── agent_state.py
│   ├── agent_tools.py
│   ├── retrieval.py
│   ├── rag.py
│   ├── query_processor.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── pdf_processor.py
│   ├── models.py
│   └── config.py
│
├── frontend/
│   └── app.py
│
├── data/
│   ├── uploads/
│   └── chroma/
│
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

# Installation

## 1. Clone the Repository

```bash
git clone https://github.com/BSaiCharan-GH/RAG-Document-Assistant.git
cd RAG-Document-Assistant
```

## 2. Create the Environment

Using Conda:

```bash
conda create -n rag-assistant python=3.11
conda activate rag-assistant
```

Or using a Python virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Environment Configuration

Create a `.env` file in the project root.

Example:

```env
GEMINI_API_KEY=your_gemini_api_key
TAVILY_API_KEY=your_tavily_api_key

EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

CANDIDATE_TOP_K=20
FINAL_TOP_K=4
RERANKER_THRESHOLD=0.0
MAX_CONTEXT_CHARS=12000

AGENT_MAX_ITERATIONS=5
AGENT_TEMPERATURE=0.1

WEB_SEARCH_MAX_RESULTS=5
WEB_SEARCH_TOPIC=general
WEB_SEARCH_DEPTH=advanced
```

Never commit the `.env` file to GitHub.

---

# Running the Application

The application consists of a FastAPI backend and Streamlit frontend.

## Start FastAPI

```powershell
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8001
```

## Start Streamlit

In another terminal:

```powershell
python -m streamlit run frontend/app.py
```

The Streamlit interface provides:

- PDF document upload
- Uploaded document management
- Question answering
- Source selection display
- Retrieval information
- Web source display
- Document source display
- Final grounded answers

---

# Example Queries

## Document Only

```text
According to the uploaded textbook, explain the transformer architecture.
```

Expected source:

```text
DOCUMENT
```

---

## Web Only

```text
What is the latest version of Python?
```

Expected source:

```text
WEB
```

---

## Hybrid

```text
According to my uploaded document, how does this technology compare with its current implementation?
```

Expected source:

```text
HYBRID
```

---

## Query Requiring Additional Retrieval

```text
Explain the advantages of the architecture discussed in the document.
```

If the initial retrieval does not provide enough evidence, the agent can refine the search before producing the answer.

---

# API

The FastAPI backend provides endpoints for document management and question answering.

Typical functionality includes:

```text
GET    /health
GET    /documents
POST   /upload
DELETE /documents/{document_id}
POST   /query
```

A query response contains the generated answer together with agent and retrieval metadata.

Example:

```json
{
  "question": "Explain the transformer architecture.",
  "answer": "...",
  "source": {
    "mode": "document",
    "reason_code": "document_specific"
  },
  "agent": {
    "iterations": 2,
    "tools_used": [
      "search_documents",
      "expand_query"
    ]
  },
  "retrieval": {
    "queries": [
      "Explain the transformer architecture"
    ],
    "candidate_count": 20,
    "final_count": 4
  },
  "retrieved_chunks": [],
  "web_sources": []
}
```

---

# Security

The application is designed to avoid exposing sensitive configuration through the frontend or source repository.

Important practices:

- API keys are stored in environment variables.
- `.env` is excluded from Git.
- Uploaded files are processed locally.
- File names are sanitized.
- Duplicate documents are detected using file hashing.
- User queries are validated using Pydantic models.
- Retrieval limits prevent unnecessarily large context windows.
- Web search results are treated as external evidence rather than trusted system instructions.

---

# Error Handling

The system handles common failures such as:

- Invalid PDF uploads
- Empty documents
- Duplicate documents
- Missing API keys
- Failed embedding generation
- Vector database errors
- Web search failures
- Empty retrieval results
- Insufficient evidence
- Agent iteration limits
- Invalid user queries

If web search fails, the application can continue using available document evidence when appropriate.

---

# Performance Considerations

The system balances retrieval quality and computational cost.

### Local Embeddings

Document embeddings are generated locally using Sentence Transformers.

### Candidate Retrieval

A larger candidate pool is retrieved before reranking.

```text
20 candidates
      ↓
Cross-Encoder
      ↓
4 final passages
```

### Context Limiting

The amount of context passed to the language model is limited to prevent unnecessarily large prompts.

### Agent Iteration Limit

The agent has a maximum number of iterations to prevent uncontrolled retrieval loops.

---

# Limitations

The system currently depends on:

- Quality of uploaded documents
- Quality of semantic retrieval
- Cross-encoder ranking quality
- Availability of the language model
- Availability and quality of web search results
- Accuracy of external web sources

Web information can change over time, and retrieved web content should therefore be interpreted according to its source and publication context.

---

# Future Improvements

Potential extensions include:

- More advanced query decomposition
- Multiple-document reasoning
- Page-level document retrieval
- Better duplicate and semantic similarity detection
- Additional web tools
- Source credibility analysis
- Improved evidence conflict resolution
- Conversation memory
- Streaming agent execution
- Authentication
- User-specific document collections
- Retrieval and answer-quality evaluation benchmarks
- Agent execution tracing
- Automated retrieval evaluation

---

# Design Principles

### 1. Retrieval Before Generation

The language model should answer using retrieved evidence rather than relying solely on its internal knowledge.

### 2. Agentic Decision Making

The system determines which information source is appropriate instead of applying a single fixed retrieval strategy to every query.

### 3. Evidence Sufficiency

Retrieval results are evaluated before the final answer is generated.

### 4. Source Transparency

Users can see whether an answer was generated from documents, web information, or both.

### 5. Iterative Retrieval

The agent can refine its search when the first retrieval attempt is insufficient.

### 6. Separation of Concerns

Document processing, retrieval, agent orchestration, web search, API handling, and frontend presentation are maintained as separate components.

---

# Project Goal

The goal of this project is to build a production-oriented agentic retrieval system that goes beyond traditional fixed-pipeline RAG.

Traditional RAG:

```text
Question
   ↓
Vector Search
   ↓
LLM
   ↓
Answer
```

Agentic RAG:

```text
Question
   ↓
Analyze
   ↓
Decide
   ↓
Retrieve
   ↓
Evaluate
   ↓
Refine if necessary
   ↓
Generate
   ↓
Provide Sources
```

The system combines private document knowledge with current web information while maintaining explicit source provenance throughout the answer-generation process.
