# Deploy the Agentic RAG Document Assistant

This deployment uses the project's existing PDF processing, embeddings, ChromaDB, retrieval/reranking, LangGraph agent, Gemini generation, and optional Tavily web-search components. The new root `streamlit_app.py` calls the existing backend services directly in-process, so Streamlit Community Cloud does not need a second public FastAPI service. The original `backend/main.py` and `frontend/app.py` remain available for local/API use.

## 1. Push the minimal deployment changes

1. Extract `RAG_Document_Assistant_Deployment.zip` into a folder, for example `C:\Users\<you>\Downloads\rag-deploy`.
2. Open PowerShell in that extracted folder (File Explorer address bar → type `powershell` → Enter).
3. If your current PowerShell policy blocks scripts, apply a temporary change for that window only:

   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
   ```

4. Run:

   ```powershell
   .\publish_to_github.ps1
   ```

5. Complete Git Credential Manager's browser sign-in if requested. The script clones the current `main`, adds the new Streamlit entrypoint and config, applies a backwards-compatible optional `persist_dir` parameter to `backend/vector_store.py`, commits the changes and pushes to `main`.
6. Confirm `GitHub sync succeeded.` and verify that `streamlit_app.py` appears at the repository root.

The script also updates the Google SDK requirement ranges to match the current LangChain Google integration. The repository originally pinned `google-genai==1.0.0`, which conflicts with current `langchain-google-genai` releases and can stop a Cloud build. It does not copy or overwrite `.env`, API keys, uploaded PDFs, vector database files, or existing frontend/backend source files besides the small backward-compatible Chroma constructor change.

## 2. Get API keys

Gemini is required for answer generation. Create a key using [Google AI Studio](https://aistudio.google.com/app/apikey). Tavily is optional for document-only use but required to enable live WEB and HYBRID search; get a key from [Tavily](https://app.tavily.com/). Free-tier availability, quotas, and pricing depend on each provider's current terms.

Never commit either key to GitHub and never paste the keys into chat.

## 3. Deploy Streamlit Community Cloud

1. Open [Streamlit Community Cloud](https://share.streamlit.io) and sign in with the GitHub account that can access this repository.
2. Click **Create app** → **Deploy an app**.
3. Set:
   - **Repository:** `BSaiCharan-GH/RAG-Document-Assistant`
   - **Branch:** `main`
   - **Main file path:** `streamlit_app.py`
4. Open **Advanced settings** before deploying and select **Python 3.11**.
5. In the **Secrets** field, enter TOML with your actual keys:

   ```toml
   GEMINI_API_KEY = "YOUR_REAL_GEMINI_API_KEY"
   GEMINI_MODEL = "gemini-3.6-flash"
   AGENT_MODEL = "gemini-3.6-flash"
   TAVILY_API_KEY = "YOUR_REAL_TAVILY_API_KEY"
   ```

   If you do not have a Tavily key yet, omit the `TAVILY_API_KEY` line. Document-only questions can still work, but live web and hybrid source retrieval will be unavailable.

6. Click **Deploy** and wait for the dependency installation and first model downloads to finish.

## 4. Smoke test the hosted app

1. Upload a small, text-based PDF you have permission to use.
2. Confirm the document appears in **Documents in this session** and that a non-zero chunk count is reported.
3. Ask a specific question answered by the PDF. Confirm the app reports the selected source strategy and shows supporting document passages with page numbers and retrieval/reranker scores.
4. With `TAVILY_API_KEY` configured, try a current-information query and a hybrid question referring to the uploaded document and current context.
5. Test removing a document from the session.

Scanned/image-only PDFs are not OCR'd by the current project; the uploader will tell you when no selectable text is found. Keep this limitation visible in demos rather than implying OCR support.

## Storage and privacy notes

- Each Streamlit browser session receives its own random workspace directory and Chroma database to avoid pooling uploaded documents across visitors.
- The demo limits each session to 5 PDFs, 60 MB of PDF files total, and 20 queries. These per-session caps are basic guardrails, not complete protection against abuse.
- Streamlit Community Cloud's local storage is temporary; documents and vectors may disappear after an app restart or redeploy.
- This is a portfolio demo, not a hardened multi-user document-management service. Do not upload confidential documents or documents you are not authorized to process. A public app can consume the API quotas associated with your configured Gemini and Tavily keys; review provider quotas and consider making the app private during setup/testing.
- The existing FastAPI server and frontend remain in the repository; the deployed app uses the same backend service classes in-process to avoid a second hosting service.
