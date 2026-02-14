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
  - Uploaded document text (if provided) is chunked and semantically ranked with the same embedding model used in Streamlit (`all-MiniLM-L6-v2`).
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
  - `HUGGINGFACE_API_TOKEN`, `GROQ_API_KEY`
  - `RAG_MODEL_ID`, `RAG_TEMPERATURE`, `RAG_MAX_TOKENS`
  - `UPLOAD_CHUNK_SIZE`, `UPLOAD_CHUNK_OVERLAP`
