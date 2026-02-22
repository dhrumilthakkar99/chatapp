# RAG Service (Python)

`apps/rag-service` is now the production Python intelligence engine used by `apps/api`.
It contains migrated retrieval + LLM routing behavior from the Streamlit reference (`app.py`).

## Run locally

```bash
python -m pip install -r requirements.txt
uvicorn app.main:app --reload --port 8002 --app-dir .
```

## API contract

- `POST /query` accepts:
  - `sessionId`, `message`, `history`, `topK`
  - optional `document` object (`documentName`, `documentKind`, `documentText`)
- returns:
  - `answer`
  - `retrievedChunks[]` with `id`, `page`, `chunkType`, `text`
  - optional span mapping fields: `startOffset`, `endOffset`, `sourceDocument`, `score`
  - `citations[]` aligned to returned chunks

## Runtime behavior

- Primary retrieval:
  - Uploaded document text (if provided) is chunked and ranked.
  - Default mode is `lexical` for low-memory/free-tier reliability.
  - Optional mode `semantic` uses `all-MiniLM-L6-v2` embeddings (higher memory).
- Secondary retrieval:
  - Qdrant if configured, else FAISS local fallback (`faiss_store/`).
- Answer generation:
  - HF Router->Groq primary route, Groq fallback.
  - Local guarded fallback answer if LLM route is unavailable.

## Required/optional env

- `RAG_SERVICE_URL` (set in API, points to this service)
- Optional:
  - `QDRANT_URL`, `QDRANT_API_KEY`, `QDRANT_COLLECTION`
  - `KB_BACKEND` (`qdrant` or `faiss`)
  - `RAG_RETRIEVAL_MODE` (`lexical` or `semantic`, default `lexical`)
  - `HUGGINGFACE_API_TOKEN`, `GROQ_API_KEY`
  - `RAG_MODEL_ID`, `RAG_TEMPERATURE`, `RAG_MAX_TOKENS`
  - `UPLOAD_CHUNK_SIZE`, `UPLOAD_CHUNK_OVERLAP`

## Deploy on Hugging Face Spaces (free CPU)

1. Create a new Space with SDK = `Docker`.
2. Point the Space to `apps/rag-service`.
3. The included `Dockerfile` exposes FastAPI on port `7860`.
4. Add secrets:
   - `QDRANT_URL`, `QDRANT_API_KEY`, `HUGGINGFACE_API_TOKEN`, `GROQ_API_KEY`
5. Add variables:
   - `KB_BACKEND=qdrant`
   - `RAG_RETRIEVAL_MODE=lexical`
   - `QDRANT_COLLECTION=doc_kb`
