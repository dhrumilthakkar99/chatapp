from __future__ import annotations

import json
import math
import os
import re
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI
from openai import OpenAI
from pydantic import BaseModel, Field

from langchain_community.vectorstores import FAISS
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.models import Distance, VectorParams


EMBED_DIM = 384
FAISS_DIR = Path(os.getenv("FAISS_DIR", "faiss_store"))
PAGE_RE = re.compile(r"\[PAGE\s+(\d+)\]", re.IGNORECASE)
FIGURE_REF_RE = re.compile(r"\b(fig(?:ure)?\.?)\s*(\d+)\b", re.IGNORECASE)
TABLE_REF_RE = re.compile(r"\b(tab(?:le)?\.?)\s*(\d+)\b", re.IGNORECASE)
CITATION_REF_RE = re.compile(r"\[(S\d+)\]")

VISUAL_TYPES = {
    "figure_explain",
    "figure_ocr",
    "figure_caption",
    "table_explain",
    "table_ocr",
    "table_caption",
    "page_ocr",
    "table_pdfplumber",
    "table_unstructured",
    "tesseract_ocr",
}

_VECTORSTORE: Optional[Any] = None
_EMBEDDINGS: Optional[Any] = None

STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "your",
    "about",
    "what",
    "which",
    "where",
    "when",
    "how",
    "why",
    "does",
    "did",
    "are",
    "is",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "can",
    "could",
    "should",
    "would",
    "may",
    "might",
    "will",
    "shall",
    "a",
    "an",
    "of",
    "to",
    "in",
    "on",
    "at",
    "by",
    "as",
    "it",
    "its",
    "or",
    "if",
    "but",
    "not",
    "we",
    "our",
    "you",
    "they",
    "their",
    "them",
}

DOC_GROUNDED_RE = re.compile(
    r"\b(document|doc|pdf|page|citation|snippet|source|context|table|figure|selected)\b",
    flags=re.IGNORECASE,
)
DEFAULT_MODEL_ID = "llama-3.1-8b-instant"
DEFAULT_MODEL_CANDIDATES = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-20b",
]


class HistoryMessage(BaseModel):
    role: str
    content: str


class DocumentContext(BaseModel):
    documentName: Optional[str] = None
    documentKind: Optional[str] = None
    documentText: Optional[str] = None


class QueryRequest(BaseModel):
    sessionId: str
    message: str
    userId: Optional[str] = None
    history: list[HistoryMessage] = Field(default_factory=list)
    topK: int = 6
    document: Optional[DocumentContext] = None


class RetrievedChunk(BaseModel):
    id: str
    page: int
    chunkType: str
    text: str
    startOffset: Optional[int] = None
    endOffset: Optional[int] = None
    sourceDocument: Optional[str] = None
    score: Optional[float] = None


class Citation(BaseModel):
    id: str
    page: int
    chunkType: str
    text: str
    startOffset: Optional[int] = None
    endOffset: Optional[int] = None
    sourceDocument: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    retrievedChunks: list[RetrievedChunk]
    citations: list[Citation]


app = FastAPI(title="chatqna-rag-service", version="0.2.0")


def get_secret(name: str, default: Optional[str] = None) -> Optional[str]:
    return os.getenv(name, default)


def get_bool_secret(name: str, default: bool = False) -> bool:
    raw = (get_secret(name, str(default)) or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def normalize_text(text: str) -> str:
    return (text or "").strip()


def get_embeddings() -> Any:
    """
    Lazy-load sentence-transformers stack to keep startup memory low.
    """
    global _EMBEDDINGS
    if _EMBEDDINGS is None:
        from langchain_huggingface import HuggingFaceEmbeddings

        _EMBEDDINGS = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return _EMBEDDINGS


def get_retrieval_mode() -> str:
    mode = (get_secret("RAG_RETRIEVAL_MODE", "semantic") or "semantic").strip().lower()
    return mode if mode in {"lexical", "semantic"} else "semantic"


def get_rerank_enabled() -> bool:
    return get_bool_secret("RAG_LLM_RERANK", True)


def get_rerank_candidate_k(top_k: int) -> int:
    cfg = int(get_secret("RAG_RERANK_CANDIDATES", "8") or "8")
    cfg = max(3, min(cfg, 16))
    return max(int(top_k), cfg)


def hf_routed_model(model_id: str) -> str:
    return model_id if ":" in model_id else f"{model_id}:groq"


def get_model_candidates(model_id: str) -> list[str]:
    preferred = normalize_text(model_id) or DEFAULT_MODEL_ID
    raw = (get_secret("RAG_MODEL_CANDIDATES", "") or "").strip()
    if raw:
        candidates = [normalize_text(part) for part in raw.split(",")]
        candidates = [c for c in candidates if c]
    else:
        candidates = [preferred] + [m for m in DEFAULT_MODEL_CANDIDATES if m != preferred]

    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = candidate.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped or [DEFAULT_MODEL_ID]


def openai_chat(
    client: OpenAI,
    model: str,
    messages: list[dict[str, Any]],
    temperature: float,
    max_tokens: int,
) -> tuple[str, Any]:
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        n=1,
    )
    content = resp.choices[0].message.content if resp and resp.choices else ""
    return content or "", resp


def llm_chat_with_fallback(
    model_id: str,
    messages: list[dict[str, Any]],
    temperature: float,
    max_tokens: int,
    hf_token: Optional[str],
    groq_key: Optional[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "content": "",
        "primary_used": False,
        "raw": None,
        "used_model": None,
        "error_primary": None,
        "error_fallback": None,
    }
    candidate_models = get_model_candidates(model_id)

    if hf_token:
        primary_errors: list[str] = []
        try:
            hf_client = OpenAI(base_url="https://router.huggingface.co/v1", api_key=hf_token)
            for candidate in candidate_models:
                try:
                    routed = hf_routed_model(candidate)
                    content, raw = openai_chat(hf_client, routed, messages, temperature, max_tokens)
                    if normalize_text(content):
                        result.update(
                            {
                                "content": content,
                                "primary_used": True,
                                "raw": raw,
                                "used_model": candidate,
                            }
                        )
                        return result
                    primary_errors.append(f"{candidate}: empty content")
                except Exception as exc:  # pragma: no cover
                    primary_errors.append(f"{candidate}: {type(exc).__name__}: {exc}")
        except Exception as exc:  # pragma: no cover
            primary_errors.append(f"hf_client_init: {type(exc).__name__}: {exc}")
        result["error_primary"] = " | ".join(primary_errors[:4]) or "Primary returned empty content."
    else:
        result["error_primary"] = "Missing HUGGINGFACE_API_TOKEN"

    if not groq_key:
        result["error_fallback"] = "Missing GROQ_API_KEY (fallback not available)"
        return result

    fallback_errors: list[str] = []
    try:
        groq_client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=groq_key)
        for candidate in candidate_models:
            try:
                content, raw = openai_chat(groq_client, candidate, messages, temperature, max_tokens)
                if normalize_text(content):
                    result.update({"content": content, "raw": raw, "used_model": candidate})
                    return result
                fallback_errors.append(f"{candidate}: empty content")
            except Exception as exc:  # pragma: no cover
                fallback_errors.append(f"{candidate}: {type(exc).__name__}: {exc}")
    except Exception as exc:  # pragma: no cover
        fallback_errors.append(f"groq_client_init: {type(exc).__name__}: {exc}")
    result["error_fallback"] = " | ".join(fallback_errors[:4]) or "Fallback returned empty content."
    return result


def load_faiss(embeddings: Any, path: Path = FAISS_DIR) -> Optional[FAISS]:
    if (path / "index.faiss").exists() and (path / "index.pkl").exists():
        return FAISS.load_local(str(path), embeddings, allow_dangerous_deserialization=True)
    return None


def init_qdrant_vectorstore() -> tuple[Optional[QdrantVectorStore], Optional[str]]:
    url = get_secret("QDRANT_URL")
    api_key = get_secret("QDRANT_API_KEY")
    collection_name = get_secret("QDRANT_COLLECTION", "doc_kb")
    if not url or not api_key:
        return None, "Missing QDRANT_URL or QDRANT_API_KEY"

    try:
        client = QdrantClient(url=url, api_key=api_key)
        try:
            client.get_collection(collection_name=collection_name)
        except Exception:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
            )
        store = QdrantVectorStore(client=client, collection_name=collection_name, embedding=get_embeddings())
        return store, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def ensure_vectorstore() -> Optional[Any]:
    global _VECTORSTORE
    if _VECTORSTORE is not None:
        return _VECTORSTORE

    backend = (get_secret("KB_BACKEND", "qdrant") or "qdrant").lower()
    if backend.startswith("qdrant"):
        store, _ = init_qdrant_vectorstore()
        if store is not None:
            _VECTORSTORE = store
            return _VECTORSTORE

    _VECTORSTORE = load_faiss(get_embeddings())
    return _VECTORSTORE


def qdrant_filter_for_chunk_types(types: list[str]) -> qmodels.Filter:
    should: list[qmodels.FieldCondition] = []
    for path in ("metadata.chunk_type", "chunk_type"):
        should.append(qmodels.FieldCondition(key=path, match=qmodels.MatchAny(any=types)))
    return qmodels.Filter(should=should)


def is_visual_question(question: str) -> bool:
    return bool(re.search(r"\b(fig|figure|diagram|chart|graph|image|table|tab)\b", question or "", flags=re.I))


def extract_ref(question: str) -> Optional[str]:
    m = FIGURE_REF_RE.search(question or "")
    if m:
        return f"figure {m.group(2)}"
    m = TABLE_REF_RE.search(question or "")
    if m:
        return f"table {m.group(2)}"
    return None


def retrieve_docs(vectorstore: Any, query: str, k: int, prefer_visual: bool) -> list[Any]:
    docs_all: list[Any] = []
    if prefer_visual and isinstance(vectorstore, QdrantVectorStore):
        try:
            flt = qdrant_filter_for_chunk_types(list(VISUAL_TYPES))
            docs_vis = vectorstore.similarity_search(query, k=min(12, max(k, 10)), filter=flt)
            docs_all.extend(docs_vis)
        except Exception:
            pass

    try:
        docs_gen = vectorstore.similarity_search(query, k=max(k, 6))
        docs_all.extend(docs_gen)
    except Exception:
        return []

    seen: set[str] = set()
    uniq: list[Any] = []
    for doc in docs_all:
        text = getattr(doc, "page_content", "") or ""
        key = text.strip()[:2000]
        if key and key not in seen:
            seen.add(key)
            uniq.append(doc)

    if prefer_visual and not isinstance(vectorstore, QdrantVectorStore):
        vis_first: list[Any] = []
        rest: list[Any] = []
        for doc in uniq:
            md = getattr(doc, "metadata", {}) or {}
            chunk_type = (md.get("chunk_type") or "").lower()
            if chunk_type in VISUAL_TYPES:
                vis_first.append(doc)
            else:
                rest.append(doc)
        uniq = vis_first + rest

    return uniq[: max(k, 10) if prefer_visual else k]


def infer_page_from_text(text: str) -> int:
    match = PAGE_RE.search(text or "")
    if match:
        try:
            return max(1, int(match.group(1)))
        except ValueError:
            return 1
    return 1


def infer_page_from_offset(text: str, offset: int) -> int:
    if not text:
        return 1
    before = text[: max(0, min(len(text), offset))]
    matches = list(PAGE_RE.finditer(before))
    if not matches:
        return 1
    try:
        return max(1, int(matches[-1].group(1)))
    except ValueError:
        return 1


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    na = math.sqrt(dot(a, a))
    nb = math.sqrt(dot(b, b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot(a, b) / (na * nb)


def tokenize_retrieval(text: str) -> list[str]:
    toks = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-_]{1,}", text or "")
    return [t.lower() for t in toks if t.lower() not in STOPWORDS]


def lexical_overlap_score(query: str, text: str) -> float:
    q_tokens = tokenize_retrieval(query)
    t_tokens = tokenize_retrieval(text)
    if not q_tokens or not t_tokens:
        return 0.0
    q_count = Counter(q_tokens)
    t_count = Counter(t_tokens)
    overlap = sum(min(cnt, t_count.get(tok, 0)) for tok, cnt in q_count.items())
    if overlap <= 0:
        return 0.0
    # Lightweight score favoring chunks with better query coverage.
    return overlap / float(len(q_tokens) + 0.35 * len(t_tokens))


def split_text_with_offsets(text: str, chunk_size: int, chunk_overlap: int) -> list[dict[str, Any]]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    parts = splitter.split_text(text or "")
    chunks: list[dict[str, Any]] = []
    cursor = 0
    for part in parts:
        probe = part[:220]
        if not probe:
            continue
        idx = text.find(probe, max(0, cursor - chunk_overlap - 500))
        if idx < 0:
            idx = text.find(probe)
        if idx < 0:
            continue
        end = min(len(text), idx + len(part))
        cursor = end
        chunks.append(
            {
                "text": part,
                "startOffset": idx,
                "endOffset": end,
                "page": infer_page_from_offset(text, idx),
                "chunkType": "main_text",
            }
        )
    return chunks


def retrieve_from_uploaded_document(payload: QueryRequest) -> list[dict[str, Any]]:
    document = payload.document
    document_text = (document.documentText if document else "") or ""
    if not document_text.strip():
        return []

    chunk_size = int(get_secret("UPLOAD_CHUNK_SIZE", "1000") or "1000")
    chunk_overlap = int(get_secret("UPLOAD_CHUNK_OVERLAP", "200") or "200")
    chunks = split_text_with_offsets(document_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunks:
        return []

    ranked: list[dict[str, Any]] = []
    if get_retrieval_mode() == "semantic":
        embeddings = get_embeddings()
        query_vec = embeddings.embed_query(payload.message)
        doc_vecs = embeddings.embed_documents([c["text"] for c in chunks])
        for chunk, vec in zip(chunks, doc_vecs):
            score = cosine_similarity(query_vec, vec)
            ranked.append({**chunk, "score": float(score)})
    else:
        for chunk in chunks:
            score = lexical_overlap_score(payload.message, str(chunk.get("text") or ""))
            ranked.append({**chunk, "score": float(score)})

    ranked.sort(key=lambda c: c["score"], reverse=True)
    candidate_k = get_rerank_candidate_k(payload.topK) if get_rerank_enabled() else int(payload.topK)
    top_k = max(1, min(candidate_k, len(ranked)))
    out: list[dict[str, Any]] = []
    for idx, item in enumerate(ranked[:top_k], start=1):
        out.append(
            {
                "id": f"S{idx}",
                "page": int(item["page"]),
                "chunkType": str(item.get("chunkType") or "main_text"),
                "text": str(item["text"]),
                "startOffset": int(item["startOffset"]),
                "endOffset": int(item["endOffset"]),
                "sourceDocument": document.documentName if document else None,
                "score": round(float(item.get("score", 0.0)), 5),
            }
        )
    return out


def locate_offsets_in_document(text: str, snippet: str) -> tuple[Optional[int], Optional[int]]:
    source = text or ""
    needle = (snippet or "").strip()
    if not source or not needle:
        return None, None
    probe = needle[:260]
    idx = source.lower().find(probe.lower())
    if idx < 0:
        return None, None
    return idx, min(len(source), idx + len(needle))


def retrieve_from_vectorstore(payload: QueryRequest) -> list[dict[str, Any]]:
    vectorstore = ensure_vectorstore()
    if vectorstore is None:
        return []

    prefer_visual = is_visual_question(payload.message)
    requested_k = get_rerank_candidate_k(payload.topK) if get_rerank_enabled() else int(payload.topK)
    k = max(int(requested_k), 10) if prefer_visual else int(requested_k)
    docs = retrieve_docs(vectorstore, payload.message, k=k, prefer_visual=prefer_visual)
    if not docs:
        return []

    document_text = ((payload.document.documentText if payload.document else "") or "").strip()
    source_document = payload.document.documentName if payload.document else None
    out: list[dict[str, Any]] = []
    for idx, doc in enumerate(docs, start=1):
        snippet = (getattr(doc, "page_content", "") or "").strip()
        if not snippet:
            continue
        md = getattr(doc, "metadata", {}) or {}
        page = md.get("page") or md.get("page_number") or infer_page_from_text(snippet)
        try:
            page_num = max(1, int(page))
        except Exception:
            page_num = 1

        start_offset = md.get("start_offset")
        end_offset = md.get("end_offset")
        if start_offset is None or end_offset is None:
            start_offset, end_offset = locate_offsets_in_document(document_text, snippet)

        out.append(
            {
                "id": f"S{idx}",
                "page": page_num,
                "chunkType": str(md.get("chunk_type") or "main_text"),
                "text": snippet,
                "startOffset": int(start_offset) if isinstance(start_offset, int) else None,
                "endOffset": int(end_offset) if isinstance(end_offset, int) else None,
                "sourceDocument": source_document or md.get("filename"),
                "score": None,
            }
        )
    return out


def extract_json_object(text: str) -> Optional[dict[str, Any]]:
    raw = (text or "").strip()
    if not raw:
        return None

    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else None
    except Exception:
        pass

    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw, flags=re.IGNORECASE)
    if fenced:
        block = fenced.group(1).strip()
        try:
            data = json.loads(block)
            return data if isinstance(data, dict) else None
        except Exception:
            pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        try:
            data = json.loads(raw[start : end + 1])
            return data if isinstance(data, dict) else None
        except Exception:
            return None
    return None


def llm_rerank_chunks(
    question: str,
    history: list[dict[str, str]],
    chunks: list[dict[str, Any]],
    model_id: str,
    hf_token: Optional[str],
    groq_key: Optional[str],
) -> tuple[list[dict[str, Any]], Optional[str]]:
    if not get_rerank_enabled() or len(chunks) < 2:
        return chunks, None

    candidate_limit = min(len(chunks), get_rerank_candidate_k(len(chunks)))
    candidates = chunks[:candidate_limit]
    by_id = {str(c.get("id")): c for c in candidates if c.get("id")}
    if len(by_id) < 2:
        return chunks, None

    history_tail = history[-4:] if history else []
    history_text = "\n".join(f"{m.get('role', 'user')}: {(m.get('content') or '')[:240]}" for m in history_tail)
    candidate_lines: list[str] = []
    for c in candidates:
        sid = str(c.get("id") or "")
        page = c.get("page") or 1
        ctype = c.get("chunkType") or "main_text"
        txt = (c.get("text") or "").strip().replace("\n", " ")
        if len(txt) > 800:
            txt = txt[:800] + "..."
        candidate_lines.append(f"[{sid}] page={page} type={ctype} text={txt}")

    rerank_messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "You are a retrieval ranker. Return ONLY JSON with keys: "
                "ordered_ids (array of snippet IDs sorted by relevance), "
                "needs_clarification (boolean), clarification_question (string). "
                "Do not answer the user question directly. "
                "If the user query is ambiguous relative to snippets, set needs_clarification=true and ask one concise question."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Conversation tail:\n{history_text or '(none)'}\n\n"
                f"Question:\n{question}\n\n"
                f"Candidate snippets:\n" + "\n".join(candidate_lines)
            ),
        },
    ]

    rerank_max_tokens = int(get_secret("RAG_RERANK_MAX_TOKENS", "280") or "280")
    rerank_result = llm_chat_with_fallback(
        model_id=model_id,
        messages=rerank_messages,
        temperature=0.0,
        max_tokens=max(96, min(rerank_max_tokens, 512)),
        hf_token=hf_token,
        groq_key=groq_key,
    )
    rerank_content = normalize_text(rerank_result.get("content", ""))
    parsed = extract_json_object(rerank_content)
    if not parsed:
        return chunks, None

    needs_clarification = bool(parsed.get("needs_clarification"))
    clarification_question = normalize_text(str(parsed.get("clarification_question") or ""))
    if needs_clarification and clarification_question:
        return chunks, clarification_question

    ordered_ids_raw = parsed.get("ordered_ids") or []
    if not isinstance(ordered_ids_raw, list):
        return chunks, None

    ordered_ids: list[str] = []
    for item in ordered_ids_raw:
        sid = str(item or "").strip()
        if sid and sid in by_id and sid not in ordered_ids:
            ordered_ids.append(sid)

    if not ordered_ids:
        return chunks, None

    re_ranked_candidates = [by_id[sid] for sid in ordered_ids]
    remaining_candidates = [c for c in candidates if str(c.get("id")) not in ordered_ids]
    return re_ranked_candidates + remaining_candidates + chunks[candidate_limit:], None


def build_qa_prompt_with_history(
    history: list[dict[str, str]],
    context_blocks: list[str],
    question: str,
    max_history_turns: int = 8,
) -> list[dict[str, str]]:
    msgs = [m for m in history if m.get("role") in ("user", "assistant")]
    if len(msgs) > max_history_turns * 2:
        msgs = msgs[-max_history_turns * 2 :]

    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "You are a precise assistant. Prefer answers grounded in the provided snippets. "
                "If the query is ambiguous (for example, 'explain this') and target passage is unclear, "
                "ask one concise clarifying question instead of guessing. "
                "If the answer is not present in snippets, say: 'Not found in the knowledge base.' "
                "When you use snippets, cite snippet IDs like [S1], [S2]. "
                "For TABLE questions, prioritize [TABLE_EXPLAIN], [TABLE_OCR], [TABLE_CAPTION], "
                "[TABLE_PDFPLUMBER], [TABLE_UNSTRUCTURED], [PAGE_OCR] snippets. "
                "For FIGURE questions, prioritize [FIGURE_EXPLAIN], [FIGURE_OCR], [FIGURE_CAPTION], [PAGE_OCR] snippets."
            ),
        }
    ]
    messages.extend(msgs)
    context_blob = "\n\n".join(context_blocks)
    messages.append({"role": "user", "content": f"Snippets:\n{context_blob}\n\nQuestion: {question}\nAnswer:"})
    return messages


def build_general_chat_prompt(history: list[dict[str, str]], question: str, max_history_turns: int = 8) -> list[dict[str, str]]:
    msgs = [m for m in history if m.get("role") in ("user", "assistant")]
    if len(msgs) > max_history_turns * 2:
        msgs = msgs[-max_history_turns * 2 :]
    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "You are ChatQnA, a concise and helpful assistant. "
                "Answer naturally. If user asks document-specific questions without available context, "
                "ask them to upload/select the relevant document section."
            ),
        }
    ]
    messages.extend(msgs)
    messages.append({"role": "user", "content": question})
    return messages


def is_doc_grounded_query(question: str) -> bool:
    return bool(DOC_GROUNDED_RE.search(question or ""))


def build_local_fallback_answer(chunks: list[dict[str, Any]]) -> str:
    if not chunks:
        return "No relevant content found in the knowledge base."
    lines = []
    for chunk in chunks[:3]:
        summary = (chunk.get("text") or "").strip().replace("\n", " ")
        if len(summary) > 220:
            summary = summary[:217] + "..."
        lines.append(f"- {summary} [{chunk['id']}]")
    return "Based on retrieved context:\n" + "\n".join(lines)


def build_citations(answer: str, chunks: list[dict[str, Any]]) -> list[Citation]:
    by_id = {str(c["id"]): c for c in chunks}
    ids: list[str] = []
    for sid in CITATION_REF_RE.findall(answer or ""):
        if sid in by_id and sid not in ids:
            ids.append(sid)

    if not ids:
        return []

    citations: list[Citation] = []
    for sid in ids:
        chunk = by_id.get(sid)
        if not chunk:
            continue
        citations.append(
            Citation(
                id=sid,
                page=int(chunk.get("page") or 1),
                chunkType=str(chunk.get("chunkType") or "main_text"),
                text=str(chunk.get("text") or ""),
                startOffset=chunk.get("startOffset"),
                endOffset=chunk.get("endOffset"),
                sourceDocument=chunk.get("sourceDocument"),
            )
        )
    return citations


@app.get("/health")
def health() -> dict[str, str]:
    backend = type(_VECTORSTORE).__name__ if _VECTORSTORE is not None else "lazy_uninitialized"
    return {"status": "ok", "service": "rag-service", "vectorstore": backend}


@app.post("/query", response_model=QueryResponse)
def query(payload: QueryRequest) -> QueryResponse:
    try:
        model_id = get_secret("RAG_MODEL_ID", DEFAULT_MODEL_ID) or DEFAULT_MODEL_ID
        temperature = float(get_secret("RAG_TEMPERATURE", "0.2") or "0.2")
        max_tokens = int(get_secret("RAG_MAX_TOKENS", "512") or "512")
        hf_token = get_secret("HUGGINGFACE_API_TOKEN")
        groq_key = get_secret("GROQ_API_KEY")
        history = [m.model_dump() for m in payload.history]

        retrieved = retrieve_from_uploaded_document(payload)
        if not retrieved:
            retrieved = retrieve_from_vectorstore(payload)

        if not retrieved:
            if not is_doc_grounded_query(payload.message):
                general_messages = build_general_chat_prompt(history, payload.message, max_history_turns=8)
                general = llm_chat_with_fallback(
                    model_id=model_id,
                    messages=general_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    hf_token=hf_token,
                    groq_key=groq_key,
                )
                general_answer = normalize_text(general.get("content", ""))
                if general_answer:
                    return QueryResponse(answer=general_answer, retrievedChunks=[], citations=[])
                return QueryResponse(
                    answer=(
                        "I can answer this, but the answer model is currently unavailable. "
                        "Please retry shortly."
                    ),
                    retrievedChunks=[],
                    citations=[],
                )
            return QueryResponse(answer="No relevant content found in the knowledge base.", retrievedChunks=[], citations=[])

        retrieved, clarifying_question = llm_rerank_chunks(
            question=payload.message,
            history=history,
            chunks=retrieved,
            model_id=model_id,
            hf_token=hf_token,
            groq_key=groq_key,
        )
        if clarifying_question:
            return QueryResponse(answer=clarifying_question, retrievedChunks=[], citations=[])

        retrieved = retrieved[: max(1, int(payload.topK))]
        context_blocks = [f"[{c['id']}] {c['text']}" for c in retrieved]
        messages = build_qa_prompt_with_history(history, context_blocks, payload.message, max_history_turns=8)

        result = llm_chat_with_fallback(
            model_id=model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            hf_token=hf_token,
            groq_key=groq_key,
        )
        answer = normalize_text(result.get("content", ""))
        if not answer:
            answer = build_local_fallback_answer(retrieved)
            if result.get("error_primary") or result.get("error_fallback"):
                print(
                    "llm_unavailable:",
                    {
                        "model_candidates": get_model_candidates(model_id),
                        "error_primary": result.get("error_primary"),
                        "error_fallback": result.get("error_fallback"),
                    },
                )
                answer += "\n\n(LLM unavailable right now; showing highest-signal retrieved context.)"

        if "not found in the knowledge base" in answer.lower() and context_blocks:
            retry_messages = [
                {
                    "role": "system",
                    "content": "You MUST answer using the snippets below. Do NOT say 'Not found' if any relevant content exists.",
                },
                {"role": "user", "content": f"Snippets:\n{'\n\n'.join(context_blocks)}\n\nQuestion: {payload.message}\nAnswer:"},
            ]
            retry = llm_chat_with_fallback(
                model_id=model_id,
                messages=retry_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                hf_token=hf_token,
                groq_key=groq_key,
            )
            retry_text = normalize_text(retry.get("content", ""))
            if retry_text:
                answer = retry_text

        citations = build_citations(answer, retrieved)
        response_chunks = [RetrievedChunk(**chunk) for chunk in retrieved]
        return QueryResponse(answer=answer, retrievedChunks=response_chunks, citations=citations)
    except Exception as exc:
        trace = traceback.format_exc(limit=4)
        print("query_exception:", trace)
        fallback = RetrievedChunk(
            id="S1",
            page=1,
            chunkType="main_text",
            text=f"rag-service exception: {type(exc).__name__}: {exc}",
        )
        return QueryResponse(
            answer=(
                "RAG query failed and returned a guarded fallback response. "
                "Check rag-service logs for details."
            ),
            retrievedChunks=[fallback],
            citations=[Citation(id="S1", page=1, chunkType="main_text", text=fallback.text)],
        )

