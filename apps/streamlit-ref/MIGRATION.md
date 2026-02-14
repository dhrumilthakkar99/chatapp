# Streamlit Reference + Production Scaffold

## Why both paths exist

- Root `app.py` is the reference implementation and source of truth for current RAG behavior.
- `apps/web` + `apps/api` + `apps/rag-service` is the production migration path.

## Migration order

1. Keep iterating prompts/logic in Streamlit.
2. Extract reusable RAG functions from `app.py` into `apps/rag-service`.
3. Keep Node API as orchestrator + persistence + auth boundary.
4. Keep React UI as chat-first product surface.

## Current status

- Clerk auth is integrated in `apps/web` and verified server-side in `apps/api`.
- `apps/rag-service` now serves migrated retrieval + LLM query behavior from `app.py`.
- Web highlighting now supports offset-based span mapping (`startOffset`, `endOffset`) when provided by `rag-service`.

## Auth (Magic Link / OAuth)

This scaffold is prepared for auth integration, but not locked to a vendor yet.
Recommended v1: Clerk (magic link + OAuth) and pass user identity to API.
