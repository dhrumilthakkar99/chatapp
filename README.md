# ChatQnA Monorepo

Production path:
- `apps/web`: React + Vite + Tailwind + Zustand chat/workspace UI
- `apps/api`: Node/Express orchestrator (auth/session/persistence gateway)
- `apps/rag-service`: Python FastAPI service where Streamlit RAG logic is migrated

Reference path:
- `app.py`: existing Streamlit implementation (ground truth + rapid prototyping)
- `apps/streamlit-ref/README.md`: notes for running the reference app

## Quick start

1. Install JS deps:
   - `pnpm install`
2. Run frontend + API:
   - `pnpm dev`
3. Run Python rag-service separately:
   - `pnpm dev:rag`

## Free Deploy (First Step)

- Follow `DEPLOY_FREE.md` for a free-tier rollout:
  - GitHub Pages (web)
  - Render free services (API + Postgres)
  - Hugging Face Spaces free CPU (rag-service)
  - Clerk free auth
  - Qdrant free cluster

## Auth (Clerk magic-link/OAuth)

`apps/web` and `apps/api` are wired for Clerk:

- Web sends Clerk bearer token on `/api/chat`
- API verifies token and uses Clerk `userId` as persisted identity

Set env:

- Web (`apps/web/.env.local`)
  - `VITE_CLERK_PUBLISHABLE_KEY=pk_...`
  - `VITE_API_BASE_URL=http://localhost:8787`
  - `VITE_BASE_PATH=/` (for GitHub Pages project sites use `/<repo-name>/`)
- API (`apps/api/.env`)
  - `CLERK_SECRET_KEY=sk_...`
  - `CORS_ORIGIN=http://localhost:5173`
  - `DATABASE_URL=postgres://chatqna:chatqna@localhost:5432/chatqna`
  - `RAG_SERVICE_URL=http://localhost:8002`
- rag-service (`apps/rag-service/.env` or shell env)
  - `GROQ_API_KEY=...`
  - `HUGGINGFACE_API_TOKEN=...`
  - optional `QDRANT_URL`, `QDRANT_API_KEY`, `QDRANT_COLLECTION`

## Infra

- `infra/docker-compose.yml` provides Postgres + Qdrant for local development.

## Notes

- Workspace is chat-first and contextually visible.
- `apps/rag-service` now contains migrated reusable retrieval/LLM logic from `app.py`.
- Chunk offsets (`startOffset`, `endOffset`) are returned by `rag-service` when available and used by web for span-based highlighting.

## Free deployment with GitHub Actions

Included workflows:

- `.github/workflows/ci.yml`
  - typecheck/build gates for `web` and `api`, plus Python compile checks.
- `.github/workflows/deploy-web-pages.yml`
  - deploys `apps/web/dist` to GitHub Pages (free static hosting).
- `.github/workflows/deploy-backend-hooks.yml`
  - triggers backend deploy hooks for API/rag-service providers.

Important:
- GitHub Pages serves static files only (no server-side Python/Node runtime).
- GitHub-hosted Actions jobs are time-limited (not suitable for always-on API hosting).

Recommended free-ish setup:

1. Deploy `apps/web` on GitHub Pages using `deploy-web-pages.yml`.
2. Deploy `apps/api` and `apps/rag-service` on providers with deploy hooks (for example Render/Railway/Fly plans if available), then set:
   - repo variable `VITE_API_BASE_URL`
   - repo secret `VITE_CLERK_PUBLISHABLE_KEY`
   - repo secrets `API_DEPLOY_HOOK`, `RAG_DEPLOY_HOOK`
3. Enable GitHub Pages in repository settings (`Build and deployment -> Source: GitHub Actions`).
