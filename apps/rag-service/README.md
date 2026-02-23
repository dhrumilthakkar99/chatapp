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
  - Default mode is `semantic` using `all-MiniLM-L6-v2` embeddings.
  - Optional mode `lexical` uses keyword overlap for lower compute.
  - Optional LLM reranker reorders retrieved chunks before answer generation.
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
  - `RAG_RETRIEVAL_MODE` (`semantic` or `lexical`, default `semantic`)
  - `RAG_LLM_RERANK` (`true`/`false`, default `true`)
  - `RAG_RERANK_CANDIDATES` (default `8`, max `16`)
  - `RAG_RERANK_MAX_TOKENS` (default `280`)
  - `HUGGINGFACE_API_TOKEN`, `GROQ_API_KEY`
  - `RAG_MODEL_ID` (default `llama-3.1-8b-instant`)
  - `RAG_MODEL_CANDIDATES` (comma-separated fallback model list)
  - `RAG_TEMPERATURE`, `RAG_MAX_TOKENS`
  - `UPLOAD_CHUNK_SIZE`, `UPLOAD_CHUNK_OVERLAP`

## Deploy on Hugging Face Spaces (free CPU)

1. Create a new Space with SDK = `Docker`.
2. Point the Space to `apps/rag-service`.
3. The included `Dockerfile` exposes FastAPI on port `7860`.
4. Add secrets:
   - `QDRANT_URL`, `QDRANT_API_KEY`, `HUGGINGFACE_API_TOKEN`, `GROQ_API_KEY`
5. Add variables:
   - `KB_BACKEND=qdrant`
   - `RAG_RETRIEVAL_MODE=semantic`
   - `RAG_LLM_RERANK=true`
   - `RAG_MODEL_ID=llama-3.1-8b-instant`
   - `RAG_MODEL_CANDIDATES=llama-3.1-8b-instant,llama-3.3-70b-versatile,openai/gpt-oss-20b`
   - `QDRANT_COLLECTION=doc_kb`
