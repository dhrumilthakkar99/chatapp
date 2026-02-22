# ChatQnA Free Deployment Guide

This guide deploys ChatQnA on free tiers first, then lets you scale later.

## Target architecture

- Frontend: GitHub Pages (`apps/web`)
- API: Render free web service (`apps/api`)
- RAG service: Hugging Face Spaces free CPU (`apps/rag-service`)
- Postgres: Render free Postgres
- Vectors: Qdrant Cloud free cluster
- Auth: Clerk free (magic link/OAuth)

## 1) Provision third-party services

1. Create Clerk app.
1. Enable email magic-link and OAuth providers you want.
1. Create Qdrant Cloud free cluster (1 GB) and copy URL/API key.
1. In Hugging Face, create a new **Space**:
   - SDK: `Docker`
   - Source directory: `apps/rag-service`
   - It will use `apps/rag-service/Dockerfile`.
1. Add Space secrets:
   - `QDRANT_URL`, `QDRANT_API_KEY`, `HUGGINGFACE_API_TOKEN`, `GROQ_API_KEY`
1. Set Space variables:
   - `KB_BACKEND=qdrant`
   - `RAG_RETRIEVAL_MODE=lexical` (cost/memory optimized default)
   - `QDRANT_COLLECTION=doc_kb`
1. Deploy the Space and copy URL (for example `https://<space-name>.hf.space`).
1. In Render, create a **Blueprint** from `render-api-only.yaml`.
1. Set `RAG_SERVICE_URL` on `chatqna-api` to your Space URL.

## 2) Configure Render env vars

### Hugging Face Space (`apps/rag-service`)

- `QDRANT_URL`
- `QDRANT_API_KEY`
- `HUGGINGFACE_API_TOKEN`
- `GROQ_API_KEY`

Defaults in `render.yaml` already set:
- `KB_BACKEND=qdrant`
- `RAG_RETRIEVAL_MODE=lexical`
- `QDRANT_COLLECTION=doc_kb`
- `RAG_MODEL_ID=openai/gpt-oss-20b`

### `chatqna-api`

- `CLERK_SECRET_KEY`
- `RAG_SERVICE_URL` (public URL of rag service)
- `CORS_ORIGIN` (your GitHub Pages frontend URL, set after step 3)

`DATABASE_URL` is wired automatically from Render Postgres via `render-api-only.yaml`.

## 3) Deploy frontend on GitHub Pages

The repo already includes `.github/workflows/deploy-web-pages.yml`.

In GitHub repo settings:

1. Enable Pages:
   - `Settings -> Pages -> Source: GitHub Actions`
1. Set repository **variables**:
   - `VITE_API_BASE_URL` = your Render API URL (for example `https://chatqna-api.onrender.com`)
   - `VITE_BASE_PATH`:
     - user/org site: `/`
     - project site: `/<repo-name>/`
1. Set repository **secret**:
   - `VITE_CLERK_PUBLISHABLE_KEY`

Push to `main` or manually run `Deploy Web To GitHub Pages` workflow.

## 4) Final CORS + Clerk setup

1. Set `CORS_ORIGIN` on Render API to your GitHub Pages URL:
   - `https://<username>.github.io` or `https://<username>.github.io/<repo-name>`
1. In Clerk dashboard:
   - add the same domain(s) under allowed origins/redirects.

## 5) Smoke tests

1. Open frontend URL and sign in with Clerk.
1. Send a chat message; verify response is returned.
1. Upload TXT/PDF and ask doc question; verify citations appear.
1. Click citation and verify viewer jumps/highlights.
1. Check Render logs (`chatqna-api`) and Hugging Face Space logs (`rag-service`) if any request fails.

## Known free-tier caveats

- Render free web services sleep after inactivity (cold start delays).
- Hugging Face Spaces free CPU sleeps when inactive (cold start delays).
- Free Postgres/storage quotas are limited.
- Qdrant free cluster is limited to 1 GB.
- GitHub Pages is static only.

## Scale-up path (later)

1. Upgrade Render services to paid instance types (always-on, faster cold start).
1. Move frontend from GitHub Pages to Cloudflare Pages/Vercel if desired.
1. Increase Qdrant cluster size.
1. Add caching + streaming + rate-limits on `apps/api`.
