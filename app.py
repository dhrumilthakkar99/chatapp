# app.py
"""
Ultimate Final (Table-aware + Fixed Follow-ups) — LLaVA v1.6 34B used for OCR/vision enrichment

- Qdrant KB + FAISS fallback
- HF Router->Groq primary + Groq fallback for LLM chat
- Vision/OCR enrichment: llava-hf/llava-v1.6-34b-hf via HF Inference (OpenAI compatible)
- Robust follow-up chaining: reuse last retrieved context for same figure/table reference
- Figure+Table-aware retrieval: prioritize caption/OCR/explain chunks for figure/table questions

Refs:
- LLaVA v1.6 34B prompt format is image + text prompt; model card gives template. [1](https://portkey.ai/models/groq/llama-3.1-70b-versatile)
- Image-text-to-text message format uses image + text content blocks. [3](https://console.groq.com/docs/text-chat)
- Base64 images can hit 413 Payload Too Large; compress/resize. [2](https://paravisionlab.co.in/text-summarizer/)
"""

import os
import re
import io
import base64
import time
import sqlite3
import traceback
import contextlib
import uuid
import html
import json
import inspect
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List

import streamlit as st
import streamlit.components.v1 as components
import fitz  # PyMuPDF
from PIL import Image

from openai import OpenAI

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http import models as qmodels


# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="Document Q&A (LLaVA v1.6 34B + Tables)", layout="wide")

def apply_modern_theme():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
        :root {
            --bg: #0f1115;
            --bg-2: #141a21;
            --panel: #1b222c;
            --panel-2: #202936;
            --text: #e7ecf3;
            --muted: #a7b0bd;
            --accent: #4cc9f0;
            --accent-2: #4895ef;
            --border: #2a3340;
        }
        .stApp {
            background: radial-gradient(circle at 10% 10%, #18202b, #0f1115 45%);
            color: var(--text);
            font-family: 'Space Grotesk', sans-serif;
        }
        section.main > div.block-container {
            max-width: 1320px;
            padding-top: 1rem;
            padding-bottom: 6rem;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #121821, #0f1115);
            border-right: 1px solid var(--border);
        }
        .stMarkdown, .stText, .stTextInput, .stTextArea, .stSelectbox, .stSlider, .stNumberInput, .stCaption {
            font-family: 'Space Grotesk', sans-serif;
        }
        div[data-testid="stChatMessage"] {
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 10px 12px;
            margin-bottom: 8px;
            background: rgba(255, 255, 255, 0.02);
        }
        .stButton > button {
            background: linear-gradient(135deg, #4cc9f0, #4895ef);
            border: 0;
            color: #0b0f14;
            font-weight: 600;
            border-radius: 10px;
            padding: 0.45rem 0.85rem;
        }
        .stButton > button:hover {
            filter: brightness(1.05);
        }
        .stTextArea textarea, .stTextInput input {
            background: var(--panel);
            color: var(--text);
            border: 1px solid var(--border);
            border-radius: 10px;
        }
        .stSelectbox div[data-baseweb="select"] > div {
            background: var(--panel);
            border-radius: 10px;
            border: 1px solid var(--border);
        }
        .stExpander {
            border: 1px solid var(--border);
            border-radius: 12px;
            background: var(--panel-2);
        }
        mark.hl {
            background: #f9c74f;
            color: #0b0f14;
            padding: 0 2px;
            border-radius: 4px;
        }
        .cite {
            color: var(--accent);
            border-bottom: 1px dotted var(--accent);
            text-decoration: none;
            font-weight: 600;
        }
        .cite:hover {
            color: #7cdaf7;
            border-bottom-color: #7cdaf7;
        }
        div[data-testid="stChatInput"] {
            position: sticky;
            bottom: 0;
            z-index: 40;
            background: linear-gradient(180deg, rgba(15,17,21,0), rgba(15,17,21,0.96) 22%, rgba(15,17,21,1));
            padding-top: 0.7rem;
            margin-top: 0.4rem;
        }
        div[data-testid="stChatInput"] > div {
            border: 1px solid var(--border);
            border-radius: 14px;
            background: rgba(27, 34, 44, 0.92);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

apply_modern_theme()
st.title("Document Summarization and Q&A (LLaVA v1.6 34B + Table-aware RAG)")


# ----------------------------
# Local FAISS persistence (fallback KB)
# ----------------------------
FAISS_DIR = Path("faiss_store")
FAISS_DIR.mkdir(exist_ok=True)

def save_faiss(vectorstore: FAISS, path: Path = FAISS_DIR):
    vectorstore.save_local(str(path))

def load_faiss(embeddings, path: Path = FAISS_DIR) -> Optional[FAISS]:
    if (path / "index.faiss").exists() and (path / "index.pkl").exists():
        return FAISS.load_local(str(path), embeddings, allow_dangerous_deserialization=True)
    return None


# ----------------------------
# SQLite chat memory (persistent)
# ----------------------------
CHAT_DB_PATH = "chat_history.sqlite"

def init_chat_db():
    con = sqlite3.connect(CHAT_DB_PATH)
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            ts TEXT NOT NULL
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS sandbox_cells (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            code TEXT NOT NULL,
            stdout TEXT NOT NULL,
            stderr TEXT NOT NULL,
            images_json TEXT NOT NULL,
            ts TEXT NOT NULL
        )
    """)
    con.commit()
    con.close()

def load_history(session_id: str) -> List[dict]:
    con = sqlite3.connect(CHAT_DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT role, content FROM chat_messages WHERE session_id=? ORDER BY id ASC", (session_id,))
    rows = cur.fetchall()
    con.close()
    return [{"role": r, "content": c} for (r, c) in rows]

def append_message(session_id: str, role: str, content: str):
    con = sqlite3.connect(CHAT_DB_PATH)
    cur = con.cursor()
    cur.execute(
        "INSERT INTO chat_messages (session_id, role, content, ts) VALUES (?, ?, ?, ?)",
        (session_id, role, content, datetime.utcnow().isoformat())
    )
    con.commit()
    con.close()

def clear_history(session_id: str):
    con = sqlite3.connect(CHAT_DB_PATH)
    cur = con.cursor()
    cur.execute("DELETE FROM chat_messages WHERE session_id=?", (session_id,))
    con.commit()
    con.close()

def load_sandbox_cells(session_id: str) -> List[dict]:
    con = sqlite3.connect(CHAT_DB_PATH)
    cur = con.cursor()
    cur.execute(
        "SELECT id, code, stdout, stderr, images_json, ts FROM sandbox_cells WHERE session_id=? ORDER BY id ASC",
        (session_id,),
    )
    rows = cur.fetchall()
    con.close()
    out = []
    for row in rows:
        images = []
        try:
            images = json.loads(row[4] or "[]")
        except Exception:
            images = []
        out.append({
            "id": row[0],
            "code": row[1] or "",
            "stdout": row[2] or "",
            "stderr": row[3] or "",
            "images": images,
            "ts": row[5] or "",
        })
    return out

def append_sandbox_cell(session_id: str, code: str, stdout_text: str, stderr_text: str, images: List[str]):
    con = sqlite3.connect(CHAT_DB_PATH)
    cur = con.cursor()
    cur.execute(
        "INSERT INTO sandbox_cells (session_id, code, stdout, stderr, images_json, ts) VALUES (?, ?, ?, ?, ?, ?)",
        (
            session_id,
            code or "",
            stdout_text or "",
            stderr_text or "",
            json.dumps(images or []),
            datetime.utcnow().isoformat(),
        ),
    )
    con.commit()
    con.close()

def clear_sandbox_cells(session_id: str):
    con = sqlite3.connect(CHAT_DB_PATH)
    cur = con.cursor()
    cur.execute("DELETE FROM sandbox_cells WHERE session_id=?", (session_id,))
    con.commit()
    con.close()


# ----------------------------
# Session state
# ----------------------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "kb_ready" not in st.session_state:
    st.session_state.kb_ready = False
if "text" not in st.session_state:
    st.session_state.text = ""
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "debug_raw" not in st.session_state:
    st.session_state.debug_raw = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_retrieved_docs" not in st.session_state:
    st.session_state.last_retrieved_docs = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# follow-up context cache
if "last_context_blocks" not in st.session_state:
    st.session_state.last_context_blocks = []
if "last_ref" not in st.session_state:
    st.session_state.last_ref = None  # e.g. "figure 2" or "table 3"
if "selected_snippet_id" not in st.session_state:
    st.session_state.selected_snippet_id = None
if "composer_text" not in st.session_state:
    st.session_state.composer_text = ""
if "last_selection_doc_main" not in st.session_state:
    st.session_state.last_selection_doc_main = ""
if "last_selection_doc_side" not in st.session_state:
    st.session_state.last_selection_doc_side = ""
if "last_selection_history" not in st.session_state:
    st.session_state.last_selection_history = ""
if "last_selected_text" not in st.session_state:
    st.session_state.last_selected_text = ""
if "last_selected_page" not in st.session_state:
    st.session_state.last_selected_page = None
if "selection_links" not in st.session_state:
    st.session_state.selection_links = []
if "pdf_bytes" not in st.session_state:
    st.session_state.pdf_bytes = None
if "pdf_filename" not in st.session_state:
    st.session_state.pdf_filename = ""
if "pdf_page_count" not in st.session_state:
    st.session_state.pdf_page_count = 0
if "viewer_page" not in st.session_state:
    st.session_state.viewer_page = 1
if "viewer_highlight_terms" not in st.session_state:
    st.session_state.viewer_highlight_terms = []
if "last_query_terms" not in st.session_state:
    st.session_state.last_query_terms = []
if "sandbox_code" not in st.session_state:
    st.session_state.sandbox_code = ""
if "sandbox_globals" not in st.session_state:
    st.session_state.sandbox_globals = {}
if "sandbox_cells" not in st.session_state:
    st.session_state.sandbox_cells = []
if "sandbox_last_result" not in st.session_state:
    st.session_state.sandbox_last_result = {"stdout": "", "stderr": "", "images": []}
if "clear_composer_on_render" not in st.session_state:
    st.session_state.clear_composer_on_render = False
if "pending_sandbox_code" not in st.session_state:
    st.session_state.pending_sandbox_code = None
if "workspace_visible" not in st.session_state:
    st.session_state.workspace_visible = False

SESSION_ID = st.session_state.session_id
init_chat_db()
if "history_synced" not in st.session_state:
    st.session_state.chat_history = load_history(SESSION_ID)
    st.session_state.sandbox_cells = load_sandbox_cells(SESSION_ID)
    st.session_state.history_synced = True


# ----------------------------
# Env helpers
# ----------------------------
def get_secret(name: str, default=None):
    return os.getenv(name) or st.secrets.get(name, default)

def dependency_available(module_name: str) -> bool:
    try:
        __import__(module_name)
        return True
    except Exception:
        return False

_COMPONENTS_HTML_SUPPORTS_KEY = "key" in inspect.signature(components.html).parameters

def render_html_component(html_block: str, height: int, key: Optional[str] = None):
    kwargs = {"height": height}
    if _COMPONENTS_HTML_SUPPORTS_KEY and key:
        kwargs["key"] = key
    return components.html(html_block, **kwargs)


# ----------------------------
# LLM routing: HF Router -> Groq (primary) + Groq fallback
# ----------------------------
def hf_routed_model(model_id: str) -> str:
    return model_id if ":" in model_id else f"{model_id}:groq"

def normalize_text(s: str) -> str:
    return (s or "").strip()

def openai_chat(client: OpenAI, model: str, messages, temperature: float, max_tokens: int):
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        n=1,
    )
    content = resp.choices[0].message.content if resp and resp.choices else ""
    return content, resp

def llm_chat_with_fallback(
    model_id: str,
    messages,
    temperature: float,
    max_tokens: int,
    hf_token: Optional[str],
    groq_key: Optional[str],
    debug: bool = False,
):
    result = {"content": "", "primary_used": False, "raw": None,
              "error_primary": None, "error_fallback": None}

    # Primary: HF Router -> Groq
    if hf_token:
        try:
            hf_client = OpenAI(base_url="https://router.huggingface.co/v1", api_key=hf_token)
            routed = hf_routed_model(model_id)
            content, raw = openai_chat(hf_client, routed, messages, temperature, max_tokens)
            if normalize_text(content):
                result.update({"content": content, "primary_used": True, "raw": raw})
                return result
            result["error_primary"] = "Primary returned empty content."
        except Exception as e:
            result["error_primary"] = f"{type(e).__name__}: {e}"
            if debug:
                st.code(traceback.format_exc())
    else:
        result["error_primary"] = "Missing HUGGINGFACE_API_TOKEN"

    # Fallback: Groq direct
    if not groq_key:
        result["error_fallback"] = "Missing GROQ_API_KEY (fallback not available)"
        return result

    try:
        groq_client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=groq_key)
        content, raw = openai_chat(groq_client, model_id, messages, temperature, max_tokens)
        result.update({"content": content, "raw": raw})
        return result
    except Exception as e:
        result["error_fallback"] = f"{type(e).__name__}: {e}"
        if debug:
            st.code(traceback.format_exc())
        return result


# ----------------------------
# Vision/OCR enrichment: LLaVA v1.6 34B via HF token
# ----------------------------
LLAVA_MODEL = "llava-hf/llava-v1.6-34b-hf"  # [1](https://portkey.ai/models/groq/llama-3.1-70b-versatile)

def image_to_data_url(img: Image.Image, max_bytes: int = 650_000) -> str:
    """
    Compress & downscale to reduce risk of HF 413 Payload Too Large for base64 image payloads. [2](https://paravisionlab.co.in/text-summarizer/)
    """
    img = img.convert("RGB")
    max_side = 1300
    w, h = img.size
    scale = min(1.0, max_side / float(max(w, h)))
    if scale < 1.0:
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))

    quality = 78
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    data = buf.getvalue()

    while len(data) > max_bytes and quality > 30:
        quality -= 10
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        data = buf.getvalue()

    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

def llava_vision_call(hf_token: str, img: Image.Image, prompt: str, max_tokens: int = 900, retries: int = 2) -> str:
    """
    Call LLaVA v1.6 34B as an image-text-to-text model. [1](https://portkey.ai/models/groq/llama-3.1-70b-versatile)[3](https://console.groq.com/docs/text-chat)
    """
    if not hf_token:
        return ""

    data_url = image_to_data_url(img)
    client = OpenAI(base_url="https://api-inference.huggingface.co/v1/", api_key=hf_token)

    messages = [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "text", "text": prompt},
        ],
    }]

    last_err = ""
    for _ in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=LLAVA_MODEL,
                messages=messages,
                temperature=0.0,
                max_tokens=max_tokens,
            )
            out = resp.choices[0].message.content if resp and resp.choices else ""
            return (out or "").strip()
        except Exception as e:
            last_err = str(e)
            if "413" in last_err or "Payload Too Large" in last_err:
                data_url = image_to_data_url(img, max_bytes=450_000)
                messages[0]["content"][0]["image_url"]["url"] = data_url
            time.sleep(1.0)

    return ""


# ----------------------------
# PDF extraction helpers (best-effort scripts)
# ----------------------------
_SUP_MAP = str.maketrans({"0":"⁰","1":"¹","2":"²","3":"³","4":"⁴","5":"⁵","6":"⁶","7":"⁷","8":"⁸","9":"⁹"})
_SUB_MAP = str.maketrans({"0":"₀","1":"₁","2":"₂","3":"₃","4":"₄","5":"₅","6":"₆","7":"₇","8":"₈","9":"₉"})

def to_sup(text: str) -> str:
    return text.translate(_SUP_MAP) if text else text

def to_sub(text: str) -> str:
    return text.translate(_SUB_MAP) if text else text

def extract_text_with_scripts(page) -> str:
    """
    Best-effort: superscript via flags + geometry; subscript via geometry.
    OCR enrichment below covers missed cases.
    """
    d = page.get_text("dict")
    out_lines = []
    for b in d.get("blocks", []):
        if b.get("type") != 0:
            continue
        for line in b.get("lines", []):
            line_bbox = line.get("bbox")
            spans = line.get("spans", [])
            if not spans:
                continue

            base_size = max(s.get("size", 0) for s in spans) or spans[0].get("size", 10)
            ly0, ly1 = (line_bbox[1], line_bbox[3]) if line_bbox else (None, None)

            line_text = ""
            for s in spans:
                txt = s.get("text", "")
                if not txt:
                    continue
                size = s.get("size", base_size)
                flags = s.get("flags", 0)
                sb = s.get("bbox")
                sy0, sy1 = (sb[1], sb[3]) if sb else (None, None)

                is_super = bool(flags & 1)
                if (not is_super) and ly0 is not None and sy1 is not None:
                    if size <= base_size * 0.80 and sy1 < (ly0 + (ly1 - ly0) * 0.55):
                        is_super = True

                is_sub = False
                if ly0 is not None and sy0 is not None:
                    if size <= base_size * 0.85 and sy0 > (ly0 + (ly1 - ly0) * 0.55):
                        is_sub = True

                if is_super and not is_sub:
                    conv = to_sup(txt)
                    line_text += conv if conv != txt else f"^({txt})"
                elif is_sub:
                    conv = to_sub(txt)
                    line_text += conv if conv != txt else f"_({txt})"
                else:
                    line_text += txt

            if line_text.strip():
                out_lines.append(line_text.strip())

    return "\n".join(out_lines).strip()


# ----------------------------
# Table/Figure caption extraction
# ----------------------------
FIG_CAPTION_RE = re.compile(r"^\s*(fig\.?|figure)\s*\d+[:.\-\s]", re.IGNORECASE)
TABLE_CAPTION_RE = re.compile(r"^\s*(tab\.?|table)\s*\d+[:.\-\s]", re.IGNORECASE)

def extract_fig_captions(text: str) -> List[str]:
    return [ln.strip() for ln in (text or "").splitlines() if FIG_CAPTION_RE.match(ln)]

def extract_table_captions(text: str) -> List[str]:
    return [ln.strip() for ln in (text or "").splitlines() if TABLE_CAPTION_RE.match(ln)]


# ----------------------------
# PDF ingestion with LLaVA enrichment (table-aware)
# ----------------------------
def render_page_image(page, dpi: int = 180) -> Image.Image:
    pix = page.get_pixmap(dpi=dpi)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

def extract_pdf_artifacts_with_llava(
    pdf_bytes: bytes,
    filename: str,
    hf_token: str,
    do_page_ocr: bool,
    do_image_ocr: bool,
    do_visual_explain: bool,
    page_ocr_mode: str,
) -> Tuple[str, List[str]]:
    """
    Table-aware enrichment:
    - If a page contains a Table caption, we force page OCR even in 'auto' mode to capture the table contents.
    - OCR prompts request markdown tables when tables appear.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    all_pages_text = []
    extras = []

    for pno in range(doc.page_count):
        page = doc.load_page(pno)

        # 1) Native text + captions
        page_text = extract_text_with_scripts(page)
        all_pages_text.append(f"[PAGE {pno+1}]\n{page_text or ''}".strip())

        # capture captions
        for cap in extract_fig_captions(page_text):
            extras.append(f"[FIGURE_CAPTION][PAGE {pno+1}] {cap}")
        for cap in extract_table_captions(page_text):
            extras.append(f"[TABLE_CAPTION][PAGE {pno+1}] {cap}")

        # 2) Page OCR-like transcription
        if do_page_ocr and hf_token:
            has_table_caption = any(TABLE_CAPTION_RE.match(ln) for ln in (page_text or "").splitlines())
            should_ocr = (page_ocr_mode == "all") or (len(page_text) < 60) or has_table_caption

            if should_ocr:
                img = render_page_image(page, dpi=180)
                prompt = (
                    "Transcribe ALL visible text EXACTLY. Preserve line breaks.\n"
                    "If there is ANY TABLE, output it as a MARKDOWN TABLE.\n"
                    "If there is a chart, include axis labels, legend entries, titles, annotations."
                )
                ocr = llava_vision_call(hf_token, img, prompt, max_tokens=1400, retries=2)
                if ocr:
                    # If we forced OCR due to table caption, mark it as table OCR as well
                    if has_table_caption and ("|" in ocr or "table" in ocr.lower()):
                        extras.append(f"[TABLE_OCR][PAGE {pno+1}] {ocr}")
                    else:
                        extras.append(f"[PAGE_OCR][PAGE {pno+1}] {ocr}")

        # 3) Embedded image OCR + optional explain
        if do_image_ocr and hf_token:
            try:
                img_list = page.get_images(full=True)
            except Exception:
                img_list = []

            for idx, imginfo in enumerate(img_list, start=1):
                xref = imginfo[0]
                try:
                    base = doc.extract_image(xref)
                    img_bytes = base.get("image", b"")
                    if not img_bytes or len(img_bytes) < 1500:
                        continue

                    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

                    prompt_ocr = (
                        "Transcribe ALL visible text EXACTLY.\n"
                        "If this is a TABLE, output it as a MARKDOWN TABLE.\n"
                        "Include headers and all rows you can read."
                    )
                    ocr = llava_vision_call(hf_token, pil_img, prompt_ocr, max_tokens=1000, retries=2)
                    if ocr:
                        # heuristically tag as table OCR if looks like markdown table
                        if "|" in ocr and "\n" in ocr:
                            extras.append(f"[TABLE_OCR][PAGE {pno+1}][IMG {idx}] {ocr}")
                        else:
                            extras.append(f"[FIGURE_OCR][PAGE {pno+1}][IMG {idx}] {ocr}")

                    if do_visual_explain:
                        prompt_explain = (
                            "Explain this visual content clearly.\n"
                            "If it's a TABLE: summarize what each column means, highlight key values/patterns, and the main takeaway.\n"
                            "If it's a FIGURE: describe axes/legend, key trends, and the main takeaway."
                        )
                        expl = llava_vision_call(hf_token, pil_img, prompt_explain, max_tokens=900, retries=2)
                        if expl:
                            # tag based on whether OCR looked table-like
                            if ocr and ("|" in ocr):
                                extras.append(f"[TABLE_EXPLAIN][PAGE {pno+1}][IMG {idx}] {expl}")
                            else:
                                extras.append(f"[FIGURE_EXPLAIN][PAGE {pno+1}][IMG {idx}] {expl}")

                except Exception:
                    continue

    doc.close()
    return "\n\n".join(all_pages_text).strip(), extras

def table_to_markdown(table: list) -> str:
    if not table:
        return ""
    rows = [[(c or "").strip() for c in row] for row in table if row]
    if not rows:
        return ""
    header = rows[0]
    if not any(header):
        header = [f"Col {i+1}" for i in range(len(rows[0]))]
    sep = ["---" for _ in header]
    body = rows[1:] if len(rows) > 1 else []
    out = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    for row in body:
        if len(row) < len(header):
            row = row + [""] * (len(header) - len(row))
        out.append("| " + " | ".join(row[: len(header)]) + " |")
    return "\n".join(out)

def extract_pdf_with_pdfplumber(pdf_bytes: bytes) -> Tuple[str, List[str]]:
    try:
        import pdfplumber
    except Exception:
        return "", []

    pages_text = []
    extras = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for pno, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages_text.append(f"[PAGE {pno}]\n{text.strip()}")

            tables = page.extract_tables() or []
            for t in tables:
                md = table_to_markdown(t)
                if md:
                    extras.append(f"[TABLE_PDFPLUMBER][PAGE {pno}] {md}")

    return "\n\n".join(pages_text).strip(), extras

def extract_pdf_with_unstructured(pdf_bytes: bytes) -> Tuple[str, List[str]]:
    try:
        from unstructured.partition.pdf import partition_pdf
    except Exception:
        return "", []

    pages_text = []
    extras = []
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp.flush()
        tmp_path = tmp.name

    try:
        elements = partition_pdf(
            filename=tmp_path,
            infer_table_structure=True,
            extract_images_in_pdf=False,
            strategy="hi_res",
        )
        for el in elements:
            text = str(el).strip()
            if not text:
                continue
            page_num = getattr(getattr(el, "metadata", None), "page_number", None) or "?"
            el_type = getattr(el, "category", None) or el.__class__.__name__
            if "Table" in str(el_type):
                extras.append(f"[TABLE_UNSTRUCTURED][PAGE {page_num}] {text}")
            else:
                pages_text.append(f"[UNSTRUCTURED_TEXT][PAGE {page_num}] {text}")
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    return "\n\n".join(pages_text).strip(), extras

def extract_pdf_with_tesseract(pdf_bytes: bytes, dpi: int = 200) -> List[str]:
    try:
        import pytesseract
    except Exception:
        return []

    extras = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for pno in range(doc.page_count):
        page = doc.load_page(pno)
        try:
            img = render_page_image(page, dpi=dpi)
            text = (pytesseract.image_to_string(img) or "").strip()
            if text:
                extras.append(f"[TESSERACT_OCR][PAGE {pno+1}] {text}")
        except Exception:
            continue
    doc.close()
    return extras

def extract_pdf_artifacts(
    pdf_bytes: bytes,
    filename: str,
    hf_token: str,
    do_page_ocr: bool,
    do_image_ocr: bool,
    do_visual_explain: bool,
    page_ocr_mode: str,
    use_pdfplumber: bool,
    use_unstructured: bool,
    use_tesseract: bool,
) -> Tuple[str, List[str]]:
    main_text, extras = extract_pdf_artifacts_with_llava(
        pdf_bytes=pdf_bytes,
        filename=filename,
        hf_token=hf_token,
        do_page_ocr=do_page_ocr,
        do_image_ocr=do_image_ocr,
        do_visual_explain=do_visual_explain,
        page_ocr_mode=page_ocr_mode,
    )

    if use_pdfplumber:
        alt_text, alt_extras = extract_pdf_with_pdfplumber(pdf_bytes)
        if alt_text and not main_text:
            main_text = alt_text
        extras.extend(alt_extras)

    if use_unstructured:
        u_text, u_extras = extract_pdf_with_unstructured(pdf_bytes)
        if u_text and not main_text:
            main_text = u_text
        extras.extend(u_extras)

    if use_tesseract:
        extras.extend(extract_pdf_with_tesseract(pdf_bytes))

    return main_text, extras


# ----------------------------
# KB: Qdrant init
# ----------------------------
EMBEDDINGS = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
EMBED_DIM = 384

def init_qdrant_vectorstore() -> Tuple[Optional[QdrantVectorStore], Optional[str]]:
    url = get_secret("QDRANT_URL", None)
    api_key = get_secret("QDRANT_API_KEY", None)
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
        vs = QdrantVectorStore(client=client, collection_name=collection_name, embedding=EMBEDDINGS)
        return vs, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


# ----------------------------
# Retrieval helpers (table/figure aware + follow-up reuse)
# ----------------------------
FIGURE_REF_RE = re.compile(r"\b(fig(?:ure)?\.?)\s*(\d+)\b", re.IGNORECASE)
TABLE_REF_RE = re.compile(r"\b(tab(?:le)?\.?)\s*(\d+)\b", re.IGNORECASE)

def extract_ref(q: str) -> Optional[str]:
    q = q or ""
    m = FIGURE_REF_RE.search(q)
    if m:
        return f"figure {m.group(2)}"
    m = TABLE_REF_RE.search(q)
    if m:
        return f"table {m.group(2)}"
    return None

def is_visual_question(q: str) -> bool:
    return bool(re.search(r"\b(fig|figure|diagram|chart|graph|image|table|tab)\b", q or "", flags=re.I))

# prioritize these chunk types for table/figure questions
VISUAL_TYPES = {
    "figure_explain", "figure_ocr", "figure_caption",
    "table_explain", "table_ocr", "table_caption",
    "page_ocr",
    "table_pdfplumber",
    "table_unstructured",
    "tesseract_ocr"
}

def qdrant_filter_for_chunk_types(types: List[str]) -> Optional[qmodels.Filter]:
    should = []
    for path in ["metadata.chunk_type", "chunk_type"]:
        should.append(qmodels.FieldCondition(key=path, match=qmodels.MatchAny(any=types)))
    return qmodels.Filter(should=should)

def retrieve_docs(vectorstore, query: str, k: int, prefer_visual: bool) -> List:
    docs_all: List = []

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

    # De-dup
    seen = set()
    uniq = []
    for d in docs_all:
        txt = getattr(d, "page_content", "") or ""
        key = txt.strip()[:2000]
        if key and key not in seen:
            seen.add(key)
            uniq.append(d)

    # For FAISS: push visual chunk types first if possible
    if prefer_visual and not isinstance(vectorstore, QdrantVectorStore):
        vis_first, rest = [], []
        for d in uniq:
            md = getattr(d, "metadata", {}) or {}
            ctype = (md.get("chunk_type") or "").lower()
            if ctype in VISUAL_TYPES:
                vis_first.append(d)
            else:
                rest.append(d)
        uniq = vis_first + rest

    return uniq[:max(k, 10) if prefer_visual else k]


# ----------------------------
# UI helpers (snippets, selection, composer)
# ----------------------------
PAGE_RE = re.compile(r"\[PAGE\s+(\d+)\]", re.IGNORECASE)

STOPWORDS = {
    "the","and","for","with","that","this","from","into","your","about","what","which","where",
    "when","how","why","does","did","are","is","was","were","be","been","being","have","has","had",
    "can","could","should","would","may","might","will","shall","a","an","of","to","in","on","at",
    "by","as","it","its","or","if","but","not","we","our","you","they","their","them"
}

def infer_page_from_text(text: str) -> Optional[str]:
    m = PAGE_RE.search(text or "")
    return m.group(1) if m else None

def build_snippet_records(items: list) -> list[dict]:
    records: list[dict] = []
    for idx, item in enumerate(items or [], start=1):
        if isinstance(item, dict):
            text = item.get("text") or item.get("page_content") or ""
            md = item.get("metadata") or {}
            rec_id = item.get("id") or f"S{idx}"
        else:
            text = getattr(item, "page_content", None)
            if text is None:
                text = str(item)
            md = getattr(item, "metadata", {}) or {}
            rec_id = f"S{idx}"
        page = md.get("page") or md.get("page_number") or infer_page_from_text(text)
        records.append({
            "id": rec_id,
            "text": text,
            "metadata": md,
            "chunk_type": (md.get("chunk_type") or "").lower(),
            "page": page,
        })
    return records

def build_snippet_label(rec: dict) -> str:
    ctype = rec.get("chunk_type") or "snippet"
    page = rec.get("page")
    parts = [rec.get("id", ""), ctype]
    if page:
        parts.append(f"page {page}")
    return " | ".join([p for p in parts if p])

def extract_query_terms(q: str, limit: int = 8) -> list[str]:
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-\_]{2,}", q or "")
    terms = []
    for t in tokens:
        tl = t.lower()
        if tl in STOPWORDS:
            continue
        if tl not in terms:
            terms.append(t)
        if len(terms) >= limit:
            break
    return terms

def highlight_html(text: str, terms: list[str]) -> str:
    safe = html.escape(text or "")
    if not terms:
        return safe
    out = safe
    for term in terms:
        if not term:
            continue
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        out = pattern.sub(lambda m: f"<mark class=\"hl\">{m.group(0)}</mark>", out)
    return out

def get_selected_snippet(records: list[dict]) -> Optional[dict]:
    if not records:
        return None
    ids = [r["id"] for r in records]
    if st.session_state.selected_snippet_id not in ids:
        st.session_state.selected_snippet_id = ids[0]
    selected_id = st.session_state.selected_snippet_id
    for r in records:
        if r["id"] == selected_id:
            return r
    return records[0]

def build_history_view_text(history: list[dict], max_messages: int = 20) -> str:
    msgs = [m for m in history if m.get("role") in ("user", "assistant")]
    if not msgs:
        return ""
    msgs = msgs[-max_messages:]
    lines = []
    for m in msgs:
        who = "You" if m.get("role") == "user" else "Assistant"
        lines.append(f"{who}: {m.get('content', '')}")
    return "\n\n".join(lines)

def render_selectable_text(text: str, key: str, height: int = 220, button_label: str = "Use selection", highlight_terms: Optional[list[str]] = None):
    safe_text = highlight_html(text or "", highlight_terms or [])
    html_block = f"""
    <style>
      mark.hl {{
        background: #f9c74f;
        color: #0b0f14;
        padding: 0 2px;
        border-radius: 4px;
      }}
    </style>
    <div style="font-family: 'IBM Plex Mono', monospace; font-size: 12.5px; color: #e7ecf3;">
      <div id="box" style="white-space: pre-wrap; background: #1b222c; padding: 10px; border: 1px solid #2a3340; border-radius: 10px; height: {height - 58}px; overflow-y: auto;">
        {safe_text}
      </div>
      <div style="margin-top: 8px; display: flex; gap: 8px;">
        <button id="btn" style="background: #4cc9f0; color: #0b0f14; border: 0; border-radius: 8px; padding: 6px 10px; font-weight: 600; cursor: pointer;">
          {button_label}
        </button>
        <button id="copy" style="background: #202936; color: #e7ecf3; border: 1px solid #2a3340; border-radius: 8px; padding: 6px 10px; cursor: pointer;">
          Copy all
        </button>
      </div>
    </div>
    <script>
      const box = document.getElementById('box');
      const btn = document.getElementById('btn');
      const copy = document.getElementById('copy');
      btn.onclick = () => {{
        const sel = window.getSelection().toString();
        const value = sel || box.innerText || "";
        Streamlit.setComponentValue(value);
      }};
      copy.onclick = () => {{
        const text = box.innerText || "";
        navigator.clipboard.writeText(text);
      }};
    </script>
    """
    return render_html_component(html_block, height=height, key=key)

def find_page_for_text(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None
    hay = st.session_state.text or ""
    needle = (text or "").strip()
    if not hay or not needle:
        return None
    needle = needle[:120]
    idx = hay.lower().find(needle.lower())
    if idx == -1:
        return None
    before = hay[:idx]
    matches = list(PAGE_RE.finditer(before))
    if matches:
        return matches[-1].group(1)
    return None

def add_selection_link(text: str, page: Optional[str], source: str):
    if not text:
        return
    page_final = page or find_page_for_text(text) or "1"
    entry = {
        "id": f"L{len(st.session_state.selection_links) + 1}",
        "text": text.strip(),
        "page": str(page_final),
        "source": source,
        "ts": datetime.utcnow().isoformat(),
    }
    key = f"{entry['page']}::{entry['text'][:80].lower()}"
    existing = {f"{e.get('page')}::{(e.get('text') or '')[:80].lower()}" for e in st.session_state.selection_links}
    if key not in existing:
        st.session_state.selection_links.append(entry)

def selection_to_terms(text: str, limit: int = 6) -> list[str]:
    terms = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-\_]{2,}", text or "")
    return terms[:limit]

def render_selection_links(selections: list[dict], key: str = "selection_links"):
    if not selections:
        return None
    items = []
    for sel in selections[-8:]:
        label = (sel.get("text") or "").strip().replace("\n", " ")
        if len(label) > 60:
            label = label[:57] + "..."
        page = sel.get("page") or "1"
        items.append({"label": label or "Selection", "page": page, "text": sel.get("text") or ""})
    data_json = json.dumps(items)
    html_block = f"""
    <div style="font-family: 'Space Grotesk', sans-serif; font-size: 13px;">
      <div id="linkList"></div>
    </div>
    <script>
      const items = {data_json};
      const root = document.getElementById('linkList');
      root.innerHTML = items.map((it, idx) => {{
        const page = it.page || '1';
        const label = it.label || 'Selection';
        return `<div style="margin-bottom: 6px;">
          <a href="#" data-page="${{page}}" data-text="${{encodeURIComponent(it.text || '')}}"
             style="color:#4cc9f0; text-decoration:none;">${{label}} (p.${{page}})</a>
        </div>`;
      }}).join('');
      root.querySelectorAll('a').forEach(a => {{
        a.addEventListener('click', (e) => {{
          e.preventDefault();
          const page = a.getAttribute('data-page') || '1';
          const text = decodeURIComponent(a.getAttribute('data-text') || '');
          Streamlit.setComponentValue(JSON.stringify({{page, text}}));
        }});
      }});
    </script>
    """
    return render_html_component(html_block, height=min(220, 42 + len(items) * 24), key=key)

def render_pdf_viewer(pdf_bytes: bytes, page: int, highlight_terms: Optional[list[str]], height: int = 640, key: str = "pdf_viewer"):
    if not pdf_bytes:
        return None
    b64 = base64.b64encode(pdf_bytes).decode("utf-8")
    terms_json = json.dumps(highlight_terms or [])
    html_block = f"""
    <style>
      .pdf-wrap {{
        height: {height}px;
        overflow: auto;
        background: #11151b;
        border: 1px solid #2a3340;
        border-radius: 12px;
        padding: 8px;
      }}
      .pdf-page {{
        margin: 0 auto 16px auto;
        background: #1b222c;
        box-shadow: 0 0 0 1px #2a3340;
        padding: 8px;
      }}
      .textLayer {{
        color: transparent;
      }}
      .textLayer span {{
        color: #e7ecf3;
      }}
      .hl {{
        background: #f9c74f;
        color: #0b0f14;
      }}
    </style>
    <div class="pdf-wrap" id="pdfContainer"></div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js"></script>
    <script>
      const pdfData = atob("{b64}");
      const container = document.getElementById("pdfContainer");
      const highlightTerms = {terms_json};
      const initialPage = {int(page) if page else 1};
      pdfjsLib.GlobalWorkerOptions.workerSrc = "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js";

      const loadingTask = pdfjsLib.getDocument({{data: pdfData}});
      loadingTask.promise.then(async (pdf) => {{
        const pageCount = pdf.numPages;
        const rendered = new Set();

        for (let i = 1; i <= pageCount; i++) {{
          const pageDiv = document.createElement('div');
          pageDiv.className = 'pdf-page';
          pageDiv.dataset.pageNumber = i;
          pageDiv.style.minHeight = '200px';
          container.appendChild(pageDiv);
        }}

        const renderPage = async (pageNumber, pageDiv) => {{
          if (rendered.has(pageNumber)) return;
          rendered.add(pageNumber);
          const page = await pdf.getPage(pageNumber);
          const viewport = page.getViewport({{ scale: 1.2 }});
          const canvas = document.createElement('canvas');
          const context = canvas.getContext('2d');
          canvas.width = viewport.width;
          canvas.height = viewport.height;
          pageDiv.appendChild(canvas);
          const renderContext = {{ canvasContext: context, viewport }};
          await page.render(renderContext).promise;

          const textContent = await page.getTextContent();
          const textLayerDiv = document.createElement('div');
          textLayerDiv.className = 'textLayer';
          pageDiv.appendChild(textLayerDiv);

          pdfjsLib.renderTextLayer({{
            textContent,
            container: textLayerDiv,
            viewport,
            textDivs: []
          }});

          if (highlightTerms && highlightTerms.length) {{
            const applyHighlights = () => {{
              let html = textLayerDiv.innerHTML;
              highlightTerms.forEach(term => {{
                if (!term) return;
                const re = new RegExp(term.replace(/[.*+?^${{}}()|[\\]\\\\]/g, '\\\\$&'), 'gi');
                html = html.replace(re, match => `<mark class="hl">${{match}}</mark>`);
              }});
              textLayerDiv.innerHTML = html;
            }};
            setTimeout(applyHighlights, 80);
          }}
        }};

        const observer = new IntersectionObserver((entries) => {{
          entries.forEach(async (entry) => {{
            if (entry.isIntersecting) {{
              const pageDiv = entry.target;
              const pageNumber = parseInt(pageDiv.dataset.pageNumber, 10);
              renderPage(pageNumber, pageDiv);
            }}
          }});
        }}, {{ root: container, rootMargin: '200px' }});

        container.querySelectorAll('.pdf-page').forEach(div => observer.observe(div));

        setTimeout(() => {{
          const target = container.querySelector(`.pdf-page[data-page-number="${{initialPage}}"]`);
          if (target) {{
            target.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
          }}
        }}, 400);
      }});
    </script>
    """
    return render_html_component(html_block, height=height + 20, key=key)

def render_answer_with_citations(answer: str, snippet_records: list[dict]) -> str:
    mapping = {}
    for rec in snippet_records or []:
        sid = rec.get("id") or ""
        if not sid:
            continue
        page = rec.get("page") or infer_page_from_text(rec.get("text", "")) or "?"
        preview = (rec.get("text") or "").strip().replace("\n", " ")
        if len(preview) > 160:
            preview = preview[:157] + "..."
        mapping[sid] = {"page": page, "preview": preview}

    def repl(match):
        sid = match.group(1)
        data = mapping.get(sid)
        if not data:
            return match.group(0)
        label = f"[{sid}, p.{data['page']}]"
        title = html.escape(data["preview"])
        return f"<span class=\"cite\" title=\"{title}\">{label}</span>"

    safe = html.escape(answer or "")
    safe = re.sub(r"\[(S\d+)\]", repl, safe)
    safe = safe.replace("\n", "<br/>")
    return safe

CITATION_REF_RE = re.compile(r"\[(S\d+)\]")

def extract_citation_records(answer: str, snippet_records: list[dict]) -> list[dict]:
    if not answer:
        return []
    ids = []
    for sid in CITATION_REF_RE.findall(answer or ""):
        if sid not in ids:
            ids.append(sid)

    by_id = {rec.get("id"): rec for rec in (snippet_records or []) if rec.get("id")}
    out = []
    for sid in ids:
        rec = by_id.get(sid)
        if not rec:
            continue
        page = rec.get("page") or infer_page_from_text(rec.get("text", "")) or "1"
        out.append({
            "id": sid,
            "page": str(page),
            "chunk_type": rec.get("chunk_type") or "snippet",
            "text": rec.get("text") or "",
        })
    return out

def jump_to_citation(citation: dict):
    page = citation.get("page") or "1"
    st.session_state.viewer_page = int(page) if str(page).isdigit() else 1
    terms = selection_to_terms(citation.get("text") or "", limit=6)
    if terms:
        st.session_state.viewer_highlight_terms = terms
    st.session_state.selected_snippet_id = citation.get("id")

ALLOWED_SANDBOX_IMPORTS = {
    "math", "statistics", "random", "json", "re", "datetime", "itertools",
    "functools", "collections", "numpy", "pandas", "matplotlib",
}

SAFE_SANDBOX_BUILTINS = {
    "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict, "enumerate": enumerate,
    "float": float, "int": int, "len": len, "list": list, "max": max, "min": min,
    "pow": pow, "print": print, "range": range, "reversed": reversed, "round": round,
    "set": set, "sorted": sorted, "str": str, "sum": sum, "tuple": tuple, "zip": zip,
}

def _sandbox_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = (name or "").split(".")[0]
    if root not in ALLOWED_SANDBOX_IMPORTS:
        raise ImportError(f"Import blocked in sandbox: {root}")
    return __import__(name, globals, locals, fromlist, level)

def run_python_sandbox(code: str) -> dict:
    code = (code or "").strip()
    if not code:
        return {"stdout": "", "stderr": "No code provided.", "images": []}

    env = st.session_state.sandbox_globals or {}
    env["__builtins__"] = dict(SAFE_SANDBOX_BUILTINS, __import__=_sandbox_import)

    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    images = []

    try:
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            exec(code, env, env)
    except Exception:
        stderr_buffer.write(traceback.format_exc())

    try:
        import matplotlib.pyplot as plt
        for fig_num in plt.get_fignums():
            fig = plt.figure(fig_num)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            images.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
        plt.close("all")
    except Exception:
        pass

    st.session_state.sandbox_globals = env
    return {
        "stdout": stdout_buffer.getvalue(),
        "stderr": stderr_buffer.getvalue(),
        "images": images,
    }

def extract_python_code_blocks(text: str) -> list[str]:
    blocks = re.findall(r"```python\s*(.*?)```", text or "", flags=re.IGNORECASE | re.DOTALL)
    return [b.strip() for b in blocks if (b or "").strip()]

def build_notebook_export(cells: list[dict]) -> str:
    nb_cells = []
    for cell in cells:
        code = cell.get("code") or ""
        stdout_text = cell.get("stdout") or ""
        stderr_text = cell.get("stderr") or ""
        outputs = []
        if stdout_text:
            outputs.append({"output_type": "stream", "name": "stdout", "text": stdout_text})
        if stderr_text:
            outputs.append({"output_type": "stream", "name": "stderr", "text": stderr_text})
        nb_cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": code,
            "outputs": outputs,
        })
    notebook = {
        "cells": nb_cells,
        "metadata": {"language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return json.dumps(notebook, indent=2)

def build_report_export(cells: list[dict]) -> str:
    lines = ["# Workspace Report", ""]
    for i, cell in enumerate(cells or [], start=1):
        lines.append(f"## Cell {i}")
        lines.append("")
        lines.append("```python")
        lines.append((cell.get("code") or "").rstrip())
        lines.append("```")
        if cell.get("stdout"):
            lines.append("")
            lines.append("**stdout**")
            lines.append("")
            lines.append("```text")
            lines.append((cell.get("stdout") or "").rstrip())
            lines.append("```")
        if cell.get("stderr"):
            lines.append("")
            lines.append("**stderr**")
            lines.append("")
            lines.append("```text")
            lines.append((cell.get("stderr") or "").rstrip())
            lines.append("```")
        lines.append("")
    return "\n".join(lines).strip()

def get_last_user_question(history: list[dict]) -> str:
    for msg in reversed(history or []):
        if msg.get("role") == "user":
            return msg.get("content") or ""
    return ""

def get_last_assistant_message(history: list[dict]) -> str:
    for msg in reversed(history or []):
        if msg.get("role") == "assistant":
            return msg.get("content") or ""
    return ""

def append_to_composer(text: str, label: Optional[str] = None):
    if not text:
        return
    current = st.session_state.composer_text or ""
    prefix = f"[{label}]\n" if label else ""
    if current.strip():
        st.session_state.composer_text = f"{current.rstrip()}\n\n{prefix}{text}".strip()
    else:
        st.session_state.composer_text = f"{prefix}{text}".strip()

def maybe_add_selection(selection: str, state_key: str, label: Optional[str] = None, page: Optional[str] = None, source: str = "snippet"):
    if not isinstance(selection, str):
        return
    selection = selection.strip()
    if not selection:
        return
    if st.session_state.get(state_key) == selection:
        return
    st.session_state[state_key] = selection
    st.session_state.last_selected_text = selection
    st.session_state.last_selected_page = page or find_page_for_text(selection)
    append_to_composer(selection, label=label)
    add_selection_link(selection, st.session_state.last_selected_page, source=source)


# ----------------------------
# Sidebar controls
# ----------------------------
st.sidebar.header("Processing Options")
chunk_size = st.sidebar.number_input("Chunk size", 200, 4000, 1000, 100)
chunk_overlap = st.sidebar.number_input("Chunk overlap", 0, 1000, 200, 50)
top_k = st.sidebar.number_input("Top K documents for retrieval", 1, 12, 6, 1)

st.sidebar.markdown("---")
st.sidebar.header("LLM Options")
model_id = st.sidebar.text_input("Model ID", value="openai/gpt-oss-20b")
max_tokens = st.sidebar.slider("Max output tokens", 64, 2048, 512, 64)
temperature = st.sidebar.slider("Temperature", 0.0, 1.5, 0.2, 0.1)
st.session_state.debug_raw = st.sidebar.checkbox("Debug raw responses", value=False)

st.sidebar.markdown("---")
st.sidebar.header("KB Backend")
kb_backend = st.sidebar.selectbox("Knowledge Base storage", ["Qdrant Cloud (API key)", "FAISS (Local fallback)"], index=0)

st.sidebar.markdown("---")
st.sidebar.header("External OCR / Vision")
st.sidebar.caption("Disabled by default for privacy. Enable only if you explicitly want external OCR calls.")
do_page_ocr = st.sidebar.checkbox("Page OCR-like transcription", value=False)
page_ocr_mode = st.sidebar.selectbox("Page OCR mode", ["auto", "all"], index=0)
do_image_ocr = st.sidebar.checkbox("OCR embedded images (figures/tables)", value=False)
do_visual_explain = st.sidebar.checkbox("Explain visual content (tables/figures)", value=False)

st.sidebar.markdown("---")
st.sidebar.header("Parsing Enhancements")
use_pdfplumber = st.sidebar.checkbox("Extract tables/text with pdfplumber", value=True)
use_unstructured = st.sidebar.checkbox("Use unstructured (advanced parsing)", value=False)
use_tesseract = st.sidebar.checkbox("Run Tesseract OCR fallback", value=False)

st.sidebar.markdown("---")
st.sidebar.header("Chat Memory")
max_history_turns = st.sidebar.slider("Max chat turns kept", 2, 20, 8, 1)
if st.sidebar.button("Clear chat history"):
    st.session_state.chat_history = []
    st.session_state.last_retrieved_docs = []
    st.session_state.last_context_blocks = []
    st.session_state.last_ref = None
    st.session_state.selected_snippet_id = None
    st.session_state.composer_text = ""
    st.session_state.last_selection_doc_main = ""
    st.session_state.last_selection_doc_side = ""
    st.session_state.last_selection_history = ""
    st.session_state.last_selected_text = ""
    st.session_state.last_selected_page = None
    st.session_state.selection_links = []
    st.session_state.viewer_highlight_terms = []
    st.session_state.last_query_terms = []
    st.session_state.clear_composer_on_render = False
    has_doc = bool(st.session_state.pdf_bytes) or bool((st.session_state.text or "").strip())
    st.session_state.workspace_visible = has_doc
    clear_history(SESSION_ID)
    st.toast("Chat history cleared.")

if st.sidebar.button("Clear workspace sandbox"):
    st.session_state.sandbox_code = ""
    st.session_state.sandbox_globals = {}
    st.session_state.sandbox_cells = []
    st.session_state.sandbox_last_result = {"stdout": "", "stderr": "", "images": []}
    st.session_state.pending_sandbox_code = None
    has_doc = bool(st.session_state.pdf_bytes) or bool((st.session_state.text or "").strip())
    st.session_state.workspace_visible = has_doc
    clear_sandbox_cells(SESSION_ID)
    st.toast("Workspace sandbox cleared.")

st.sidebar.markdown("---")
st.sidebar.header("Workspace")
st.sidebar.caption("Use the Q&A tab for the unified document, retrieval, and Python sandbox workspace.")


# ----------------------------
# Tokens
# ----------------------------
hf_token = get_secret("HUGGINGFACE_API_TOKEN", None)
groq_key = get_secret("GROQ_API_KEY", None)
if not hf_token and (do_page_ocr or do_image_ocr or do_visual_explain):
    st.warning("HUGGINGFACE_API_TOKEN missing. External OCR/vision is enabled but cannot run.")
if not groq_key:
    st.warning("GROQ_API_KEY missing. LLM fallback won't work if primary route fails.")


# ----------------------------
# Load KB vectorstore
# ----------------------------
def ensure_vectorstore():
    if kb_backend.startswith("Qdrant"):
        if not isinstance(st.session_state.vectorstore, QdrantVectorStore):
            vs, err = init_qdrant_vectorstore()
            if err:
                st.warning(f"Qdrant not available: {err}. Falling back to FAISS.")
                local = load_faiss(EMBEDDINGS)
                if local is not None:
                    st.session_state.vectorstore = local
                    st.session_state.kb_ready = True
                else:
                    st.session_state.vectorstore = None
                    st.session_state.kb_ready = False
            else:
                st.session_state.vectorstore = vs
                st.session_state.kb_ready = True
        else:
            st.session_state.kb_ready = True
    else:
        if not isinstance(st.session_state.vectorstore, FAISS):
            local = load_faiss(EMBEDDINGS)
            if local is not None:
                st.session_state.vectorstore = local
                st.session_state.kb_ready = True
            else:
                st.session_state.vectorstore = None
                st.session_state.kb_ready = False
        else:
            st.session_state.kb_ready = True

ensure_vectorstore()


# ----------------------------
# Prompt builders for Summary/Q&A
# ----------------------------
def build_summary_prompt(text: str) -> list[dict]:
    return [
        {"role": "system", "content": "Summarize faithfully and avoid adding facts. Return 8-12 bullet points."},
        {"role": "user", "content": f"Summarize the following content:\n\n{text}"},
    ]

def build_qa_prompt_with_history(history, context_blocks, question, max_history_turns=8):
    msgs = [m for m in history if m.get("role") in ("user", "assistant")]
    if len(msgs) > max_history_turns * 2:
        msgs = msgs[-max_history_turns * 2 :]

    messages = [{
        "role": "system",
        "content": (
            "You are a precise assistant. Answer using ONLY the provided snippets. "
            "If the answer is not present in snippets, say: 'Not found in the knowledge base.' "
            "Cite snippet IDs like [S1], [S2]. Every answer must include citations. "
            "For TABLE questions, prioritize [TABLE_EXPLAIN], [TABLE_OCR], [TABLE_CAPTION], [TABLE_PDFPLUMBER], [TABLE_UNSTRUCTURED], [PAGE_OCR] snippets. "
            "For FIGURE questions, prioritize [FIGURE_EXPLAIN], [FIGURE_OCR], [FIGURE_CAPTION], [PAGE_OCR] snippets."
        ),
    }]
    messages.extend(msgs)

    ctx = "\n\n".join(context_blocks)
    messages.append({"role": "user", "content": f"Snippets:\n{ctx}\n\nQuestion: {question}\nAnswer:"})
    return messages


# ----------------------------
# Deterministic "chat-about-chat"
# ----------------------------
_META_PATTERNS = {
    "last_user_q": [r"\b(last|previous)\s+(question|query)\b", r"\bwhat\s+did\s+i\s+ask\b", r"\bmy\s+last\s+question\b"],
    "summarize_chat": [r"\b(summarize|recap)\s+(our|the)\s+(chat|conversation)\b", r"\bwhat\s+did\s+we\s+discuss\b"],
}

CODE_INTENT_RE = re.compile(
    r"\b(code|python|script|function|class|debug|bug|refactor|algorithm|notebook|plot|pandas|numpy|sql)\b",
    re.IGNORECASE,
)

def is_code_prompt(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    if "```" in t:
        return True
    return bool(CODE_INTENT_RE.search(t))

def detect_meta_intent(q: str) -> str:
    ql = (q or "").strip()
    if not ql:
        return "none"
    for intent, pats in _META_PATTERNS.items():
        if any(re.search(p, ql, flags=re.I) for p in pats):
            return intent
    return "none"

def answer_meta_from_history(intent: str, history: list[dict]) -> str:
    msgs = [m for m in history if m.get("role") in ("user", "assistant")]
    if not msgs:
        return "We haven't chatted yet."
    msgs_excl_current = msgs[:-1] if msgs and msgs[-1]["role"] == "user" else msgs

    if intent == "last_user_q":
        for m in reversed(msgs_excl_current):
            if m["role"] == "user":
                return f"Your last question was: {m['content']}"
        return "I couldn't find your last question."

    if intent == "summarize_chat":
        last_n = msgs[-16:]
        lines = []
        for m in last_n:
            who = "You" if m["role"] == "user" else "Assistant"
            lines.append(f"{who}: {m['content']}")
        return "Here is a brief recap:\n\n" + "\n".join(lines)

    return "I'm not sure."


def handle_question(question_text: str):
    question_text = (question_text or "").strip()
    if not question_text:
        return

    if is_code_prompt(question_text):
        st.session_state.workspace_visible = True

    st.session_state.last_query_terms = extract_query_terms(question_text)
    st.session_state.viewer_highlight_terms = st.session_state.last_query_terms

    st.session_state.chat_history.append({"role": "user", "content": question_text})
    append_message(SESSION_ID, "user", question_text)

    ensure_vectorstore()
    if st.session_state.vectorstore is None:
        answer = "Knowledge base not ready. Configure Qdrant or process at least one document."
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        append_message(SESSION_ID, "assistant", answer)
        st.rerun()

    intent = detect_meta_intent(question_text)
    if intent != "none":
        answer = answer_meta_from_history(intent, st.session_state.chat_history)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        append_message(SESSION_ID, "assistant", answer)
        st.session_state.last_retrieved_docs = []
        st.session_state.last_context_blocks = []
        st.session_state.last_ref = None
        st.session_state.selected_snippet_id = None
        st.rerun()

    ref = extract_ref(question_text)
    prefer_visual = is_visual_question(question_text)

    reuse_prior = (
        ref is not None
        and ref == st.session_state.last_ref
        and len(st.session_state.last_context_blocks) > 0
    )

    k = max(int(top_k), 10) if prefer_visual else int(top_k)

    with st.spinner("Retrieving and answering..."):
        try:
            docs = retrieve_docs(st.session_state.vectorstore, question_text, k=k, prefer_visual=prefer_visual)

            if not docs and reuse_prior:
                context_blocks = st.session_state.last_context_blocks
                last_snips = build_snippet_records(st.session_state.last_retrieved_docs)
            elif not docs and not reuse_prior:
                answer = "No relevant content found in the knowledge base."
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                append_message(SESSION_ID, "assistant", answer)
                st.session_state.last_retrieved_docs = []
                st.session_state.last_context_blocks = []
                st.session_state.last_ref = ref
                st.session_state.selected_snippet_id = None
                st.rerun()
            else:
                context_blocks = []
                last_snips = []
                for i, d in enumerate(docs, start=1):
                    snippet = getattr(d, "page_content", "").strip()
                    if snippet:
                        context_blocks.append(f"[S{i}] {snippet}")
                        md = getattr(d, "metadata", {}) or {}
                        last_snips.append({
                            "id": f"S{i}",
                            "text": snippet,
                            "metadata": md,
                        })

                if reuse_prior:
                    merged = []
                    seen = set()
                    for b in (st.session_state.last_context_blocks + context_blocks):
                        key = b.strip()[:500]
                        if key and key not in seen:
                            seen.add(key)
                            merged.append(b)
                    context_blocks = merged

            messages = build_qa_prompt_with_history(
                st.session_state.chat_history,
                context_blocks,
                question_text,
                max_history_turns=max_history_turns,
            )

            result = llm_chat_with_fallback(
                model_id=model_id,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                hf_token=hf_token,
                groq_key=groq_key,
                debug=st.session_state.debug_raw,
            )

            answer = normalize_text(result["content"]) or "No answer returned."

            if (
                "not found in the knowledge base" in answer.lower()
                or answer.strip().lower() == "not found in the knowledge base."
            ) and len(context_blocks) > 0:
                retry_msgs = [
                    {"role": "system", "content": "You MUST answer using the snippets below. Do NOT say 'Not found' if any relevant content exists."},
                    {"role": "user", "content": f"Snippets:\n{'\n\n'.join(context_blocks)}\n\nQuestion: {question_text}\nAnswer:"},
                ]
                retry = llm_chat_with_fallback(
                    model_id=model_id,
                    messages=retry_msgs,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    hf_token=hf_token,
                    groq_key=groq_key,
                    debug=st.session_state.debug_raw,
                )
                answer2 = normalize_text(retry["content"])
                if answer2:
                    answer = answer2

            snips_for_answer = build_snippet_records(last_snips)
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer,
                "snippets": snips_for_answer,
            })
            append_message(SESSION_ID, "assistant", answer)

            st.session_state.last_retrieved_docs = last_snips
            st.session_state.last_context_blocks = context_blocks
            st.session_state.last_ref = ref or st.session_state.last_ref

            if last_snips:
                ids = [s.get("id") for s in last_snips if s.get("id")]
                if ids and st.session_state.selected_snippet_id not in ids:
                    st.session_state.selected_snippet_id = ids[0]
                first_page = snips_for_answer[0].get("page") if snips_for_answer else None
                if first_page:
                    st.session_state.viewer_page = int(first_page)

            st.rerun()

        except Exception:
            st.session_state.chat_history.append({"role": "assistant", "content": "Q&A failed."})
            append_message(SESSION_ID, "assistant", "Q&A failed.")
            st.session_state.last_retrieved_docs = []
            st.session_state.last_context_blocks = []
            st.session_state.selected_snippet_id = None
            st.rerun()


# ----------------------------
# Upload + Process Document
# ----------------------------
uploaded_file = st.file_uploader("Upload Document (PDF or TXT)", type=["pdf", "txt"])

if uploaded_file:
    try:
        if uploaded_file.type == "application/pdf":
            pdf_bytes = uploaded_file.getvalue()
            st.session_state.pdf_bytes = pdf_bytes
            st.session_state.pdf_filename = uploaded_file.name
            try:
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                st.session_state.pdf_page_count = doc.page_count
                doc.close()
            except Exception:
                st.session_state.pdf_page_count = 0

            if use_pdfplumber and not dependency_available("pdfplumber"):
                st.warning("pdfplumber is enabled but not installed. Disable it or install pdfplumber.")
            if use_unstructured and not dependency_available("unstructured"):
                st.warning("unstructured is enabled but not installed. Disable it or install unstructured.")
            if use_tesseract and not dependency_available("pytesseract"):
                st.warning("Tesseract OCR is enabled but pytesseract isn't installed. Disable it or install pytesseract.")

            main_text, extras = extract_pdf_artifacts(
                pdf_bytes=pdf_bytes,
                filename=uploaded_file.name,
                hf_token=hf_token or "",
                do_page_ocr=bool(do_page_ocr and hf_token),
                do_image_ocr=bool(do_image_ocr and hf_token),
                do_visual_explain=bool(do_visual_explain and hf_token),
                page_ocr_mode=page_ocr_mode,
                use_pdfplumber=use_pdfplumber,
                use_unstructured=use_unstructured,
                use_tesseract=use_tesseract,
            )
            st.session_state.text = main_text
            st.session_state._extra_texts = extras
            st.session_state.selection_links = []
            st.session_state.viewer_page = 1
            st.session_state.viewer_highlight_terms = []
        else:
            st.session_state.text = uploaded_file.read().decode("utf-8", errors="replace")
            st.session_state._extra_texts = []
            st.session_state.pdf_bytes = None
            st.session_state.pdf_filename = ""
            st.session_state.pdf_page_count = 0
            st.session_state.selection_links = []
            st.session_state.viewer_page = 1
            st.session_state.viewer_highlight_terms = []

        st.success(f"Document loaded: {len(st.session_state.text)} characters")
        st.session_state.workspace_visible = True
        if getattr(st.session_state, "_extra_texts", []):
            st.info(f"Extracted {len(st.session_state._extra_texts)} extra snippets (OCR/captions/tables/figures/advanced parsing).")
        if uploaded_file.type == "application/pdf" and not hf_token and (do_page_ocr or do_image_ocr or do_visual_explain):
            st.warning("External OCR/vision is disabled because HUGGINGFACE_API_TOKEN is missing.")
    except Exception as e:
        st.error(f"Failed to read uploaded file: {e}")

if st.button("Process Document") and st.session_state.text:
    with st.spinner("Chunking + embedding + storing in KB (incl. LLaVA table/figure enrichment)..."):
        try:
            ensure_vectorstore()
            splitter = RecursiveCharacterTextSplitter(chunk_size=int(chunk_size), chunk_overlap=int(chunk_overlap))

            chunks = splitter.split_text(st.session_state.text)
            st.session_state.chunks = chunks

            extras = getattr(st.session_state, "_extra_texts", [])
            extra_chunks = []
            for t in extras:
                if len(t) > 2500:
                    extra_chunks.extend(splitter.split_text(t))
                else:
                    extra_chunks.append(t)

            all_texts = chunks + extra_chunks

            meta_base = {
                "source": "upload",
                "filename": getattr(uploaded_file, "name", "unknown"),
                "filetype": getattr(uploaded_file, "type", "unknown"),
                "uploaded_at": datetime.utcnow().isoformat(),
            }

            metadatas = []
            for t in all_texts:
                m = dict(meta_base)
                # chunk types
                if t.startswith("[PAGE_OCR]"):
                    m["chunk_type"] = "page_ocr"
                elif t.startswith("[FIGURE_OCR]"):
                    m["chunk_type"] = "figure_ocr"
                elif t.startswith("[FIGURE_EXPLAIN]"):
                    m["chunk_type"] = "figure_explain"
                elif t.startswith("[FIGURE_CAPTION]"):
                    m["chunk_type"] = "figure_caption"
                elif t.startswith("[TABLE_OCR]"):
                    m["chunk_type"] = "table_ocr"
                elif t.startswith("[TABLE_EXPLAIN]"):
                    m["chunk_type"] = "table_explain"
                elif t.startswith("[TABLE_CAPTION]"):
                    m["chunk_type"] = "table_caption"
                elif t.startswith("[TABLE_PDFPLUMBER]"):
                    m["chunk_type"] = "table_pdfplumber"
                elif t.startswith("[TABLE_UNSTRUCTURED]"):
                    m["chunk_type"] = "table_unstructured"
                elif t.startswith("[UNSTRUCTURED_TEXT]"):
                    m["chunk_type"] = "unstructured_text"
                elif t.startswith("[TESSERACT_OCR]"):
                    m["chunk_type"] = "tesseract_ocr"
                else:
                    m["chunk_type"] = "main_text"
                metadatas.append(m)

            if st.session_state.vectorstore is None:
                st.session_state.vectorstore = FAISS.from_texts(all_texts, EMBEDDINGS, metadatas=metadatas)
                save_faiss(st.session_state.vectorstore)
                st.session_state.kb_ready = True
                st.success(f"Stored {len(all_texts)} chunks in FAISS KB.")
            else:
                st.session_state.vectorstore.add_texts(all_texts, metadatas=metadatas)
                if isinstance(st.session_state.vectorstore, FAISS):
                    save_faiss(st.session_state.vectorstore)
                st.session_state.kb_ready = True
                st.success(f"Stored {len(all_texts)} chunks in KB (incl. table/figure chunks).")

        except Exception as e:
            st.error(f"Failed to process document: {e}")
            st.exception(e)


# ----------------------------
# Tabs
# ----------------------------
tab2 = st.tabs(["Q&A (Chat)"])[0]

with st.expander("How it works / limitations"):
    st.markdown("- Upload a PDF or TXT file, then click `Process Document` to build the knowledge base.")
    st.markdown("- Retrieval uses vector similarity and table/figure-aware prioritization for visual questions.")
    st.markdown("- Answers are constrained to retrieved snippets and cited with snippet IDs and page numbers.")
    st.markdown("- Advanced parsing (pdfplumber, unstructured, Tesseract) is optional and depends on installed packages.")
    st.markdown("- OCR and vision enrichment can be slower and may incur additional API costs.")
    st.markdown("- Large PDFs may take longer to index; consider smaller chunk sizes for precise citations.")
    st.markdown("- For privacy-sensitive documents, avoid enabling external OCR/LLM services.")
    st.markdown("- Citations help traceability but depend on how well the PDF text was extracted.")


# ----------------------------
# Q&A tab - Table/figure aware + follow-up reuse + anti-false-not-found
# ----------------------------
with tab2:
    st.markdown("**ChatGPT-style mode: chat is primary. Workspace appears only for document/coding flows.**")

    if st.session_state.clear_composer_on_render:
        st.session_state.composer_text = ""
        st.session_state.clear_composer_on_render = False

    snippet_records = build_snippet_records(st.session_state.last_retrieved_docs)
    selected_record = get_selected_snippet(snippet_records)

    pending_question = None
    user_question = None

    has_document = bool(st.session_state.pdf_bytes) or bool((st.session_state.text or "").strip())
    if has_document:
        st.session_state.workspace_visible = True
    show_workspace = bool(st.session_state.workspace_visible)

    if show_workspace:
        col_chat, col_workspace = st.columns([1.65, 1.15], gap="medium")
    else:
        _pad_l, col_chat, _pad_r = st.columns([0.12, 0.76, 0.12], gap="small")
        col_workspace = None
        st.caption("Workspace is hidden. Upload a document or ask for coding help to unlock it.")

    with col_chat:
        for midx, msg in enumerate(st.session_state.chat_history):
            with st.chat_message("user" if msg["role"] == "user" else "assistant"):
                if msg.get("role") == "assistant" and msg.get("snippets"):
                    st.markdown(
                        render_answer_with_citations(msg.get("content", ""), msg.get("snippets", [])),
                        unsafe_allow_html=True,
                    )
                    citations = extract_citation_records(msg.get("content", ""), msg.get("snippets", []))
                    if citations:
                        ncols = min(4, max(1, len(citations)))
                        ccols = st.columns(ncols)
                        for cidx, cit in enumerate(citations):
                            label = f"{cit['id']} p.{cit['page']}"
                            with ccols[cidx % ncols]:
                                if st.button(label, key=f"msg_cite_{midx}_{cidx}", use_container_width=True):
                                    jump_to_citation(cit)
                                    st.session_state.workspace_visible = True
                                    st.rerun()
                else:
                    st.write(msg.get("content", ""))

        if not st.session_state.chat_history:
            st.markdown("### Ask anything about your document or your code workflow")
            st.caption("Chat stays clean by default. Advanced workspace appears only when needed.")

    if show_workspace and col_workspace is not None:
        with col_workspace:
            st.subheader("Workspace")
            ws_doc, ws_retrieval, ws_sandbox, ws_history = st.tabs(
                ["Document", "Retrieval Inspector", "Python Sandbox", "History"]
            )

            with ws_doc:
                if snippet_records and selected_record:
                    options = [r["id"] for r in snippet_records]
                    labels = {r["id"]: build_snippet_label(r) for r in snippet_records}
                    selected_id = st.selectbox(
                        "Retrieved snippets",
                        options=options,
                        index=options.index(selected_record["id"]) if selected_record["id"] in options else 0,
                        format_func=lambda x: labels.get(x, x),
                        key="workspace_snippet_select",
                    )
                    st.session_state.selected_snippet_id = selected_id
                    selected_record = get_selected_snippet(snippet_records)
                    st.caption(build_snippet_label(selected_record))
                    doc_selection = render_selectable_text(
                        selected_record.get("text", ""),
                        key="workspace_doc_view",
                        height=210,
                        button_label="Use selection",
                        highlight_terms=st.session_state.last_query_terms,
                    )
                    maybe_add_selection(
                        doc_selection,
                        "last_selection_doc_main",
                        label="Snippet",
                        page=selected_record.get("page"),
                        source="snippet",
                    )
                else:
                    st.info("Ask a question to retrieve snippets.")

                if st.session_state.pdf_bytes:
                    max_page = st.session_state.pdf_page_count or 1
                    viewer_page_input = st.number_input(
                        "Viewer page",
                        min_value=1,
                        max_value=max_page,
                        value=int(st.session_state.viewer_page or 1),
                        step=1,
                        key="workspace_viewer_page",
                    )
                    if viewer_page_input != st.session_state.viewer_page:
                        st.session_state.viewer_page = int(viewer_page_input)
                    viewer_key = f"pdf_workspace_{st.session_state.viewer_page}_{abs(hash(tuple(st.session_state.viewer_highlight_terms or []))) % 10000}"
                    render_pdf_viewer(
                        st.session_state.pdf_bytes,
                        page=int(st.session_state.viewer_page or 1),
                        highlight_terms=st.session_state.viewer_highlight_terms,
                        height=520,
                        key=viewer_key,
                    )
                else:
                    st.info("Upload a PDF to enable the document viewer.")

            with ws_retrieval:
                if snippet_records:
                    for ridx, rec in enumerate(snippet_records):
                        rid = rec.get("id") or f"S{ridx+1}"
                        rpage = rec.get("page") or "1"
                        rtype = rec.get("chunk_type") or "snippet"
                        lbl = f"{rid} | p.{rpage} | {rtype}"
                        if st.button(lbl, key=f"retrieval_link_{ridx}", use_container_width=True):
                            jump_to_citation({
                                "id": rid,
                                "page": rpage,
                                "text": rec.get("text") or "",
                                "chunk_type": rtype,
                            })
                            st.rerun()
                else:
                    st.info("No retrieval context yet.")

                st.markdown("**Selection links**")
                selection_payload = render_selection_links(st.session_state.selection_links, key="selection_links_view")
                if isinstance(selection_payload, str) and selection_payload:
                    try:
                        data = json.loads(selection_payload)
                        page = int(data.get("page") or 1)
                        st.session_state.viewer_page = page
                        terms = selection_to_terms(data.get("text") or "")
                        if terms:
                            st.session_state.viewer_highlight_terms = terms
                        st.rerun()
                    except Exception:
                        pass
                elif not st.session_state.selection_links:
                    st.info("Selections appear here as jump links.")

            with ws_sandbox:
                st.caption("Local per-session Python workspace")
                if st.session_state.pending_sandbox_code is not None:
                    st.session_state.sandbox_code = st.session_state.pending_sandbox_code
                    st.session_state.pending_sandbox_code = None
                last_assistant = get_last_assistant_message(st.session_state.chat_history)
                extracted_blocks = extract_python_code_blocks(last_assistant)
                c_seed1, c_seed2 = st.columns(2)
                with c_seed1:
                    if st.button("Use latest assistant code", use_container_width=True, key="seed_latest_code"):
                        if extracted_blocks:
                            st.session_state.sandbox_code = extracted_blocks[-1]
                        elif last_assistant:
                            st.session_state.sandbox_code = last_assistant
                        else:
                            st.info("No assistant response available yet.")
                with c_seed2:
                    if st.button("Generate analysis code", use_container_width=True, key="gen_sandbox_code"):
                        last_q = get_last_user_question(st.session_state.chat_history)
                        if not last_q:
                            st.info("Ask at least one question first.")
                        else:
                            context_chunks = []
                            for rec in snippet_records[:4]:
                                context_chunks.append(f"[{rec.get('id')}] {(rec.get('text') or '')[:1200]}")
                            gen_messages = [
                                {
                                    "role": "system",
                                    "content": (
                                        "Generate executable Python for analysis. "
                                        "Return only one ```python``` block. "
                                        "No placeholders, no prose."
                                    ),
                                },
                                {
                                    "role": "user",
                                    "content": f"Question:\n{last_q}\n\nContext:\n" + "\n\n".join(context_chunks),
                                },
                            ]
                            gen = llm_chat_with_fallback(
                                model_id=model_id,
                                messages=gen_messages,
                                temperature=min(temperature, 0.4),
                                max_tokens=max_tokens,
                                hf_token=hf_token,
                                groq_key=groq_key,
                                debug=st.session_state.debug_raw,
                            )
                            gen_text = normalize_text(gen.get("content", ""))
                            blocks = extract_python_code_blocks(gen_text)
                            st.session_state.sandbox_code = blocks[0] if blocks else gen_text

                st.text_area("Sandbox code", key="sandbox_code", height=270)
                run_col, reset_col = st.columns(2)
                run_clicked = run_col.button("Run code", use_container_width=True, key="sandbox_run_btn")
                reset_clicked = reset_col.button("Reset kernel", use_container_width=True, key="sandbox_reset_btn")

                if reset_clicked:
                    st.session_state.sandbox_globals = {}
                    st.success("Sandbox kernel reset.")

                if run_clicked:
                    result = run_python_sandbox(st.session_state.sandbox_code)
                    st.session_state.sandbox_last_result = result
                    cell = {
                        "id": len(st.session_state.sandbox_cells) + 1,
                        "code": st.session_state.sandbox_code,
                        "stdout": result.get("stdout") or "",
                        "stderr": result.get("stderr") or "",
                        "images": result.get("images") or [],
                        "ts": datetime.utcnow().isoformat(),
                    }
                    st.session_state.sandbox_cells.append(cell)
                    append_sandbox_cell(
                        SESSION_ID,
                        code=cell["code"],
                        stdout_text=cell["stdout"],
                        stderr_text=cell["stderr"],
                        images=cell["images"],
                    )

                last_result = st.session_state.sandbox_last_result or {}
                if last_result.get("stdout"):
                    st.markdown("**stdout**")
                    st.code(last_result["stdout"], language="text")
                if last_result.get("stderr"):
                    st.markdown("**stderr**")
                    st.code(last_result["stderr"], language="text")
                if last_result.get("images"):
                    st.markdown("**plots**")
                    for img_b64 in last_result.get("images", []):
                        try:
                            st.image(base64.b64decode(img_b64))
                        except Exception:
                            continue

                st.markdown("**Workspace history**")
                if st.session_state.sandbox_cells:
                    for hidx, cell in enumerate(reversed(st.session_state.sandbox_cells[-8:]), start=1):
                        with st.expander(f"Cell {len(st.session_state.sandbox_cells) - hidx + 1}"):
                            st.code(cell.get("code") or "", language="python")
                            if cell.get("stdout"):
                                st.code(cell.get("stdout") or "", language="text")
                            if cell.get("stderr"):
                                st.code(cell.get("stderr") or "", language="text")
                            if st.button("Load into editor", key=f"load_cell_{hidx}", use_container_width=True):
                                st.session_state.pending_sandbox_code = cell.get("code") or ""
                                st.rerun()
                else:
                    st.info("No sandbox runs yet.")

                notebook_payload = build_notebook_export(st.session_state.sandbox_cells)
                report_payload = build_report_export(st.session_state.sandbox_cells)
                ex1, ex2 = st.columns(2)
                ex1.download_button(
                    "Export notebook",
                    data=notebook_payload,
                    file_name="workspace.ipynb",
                    mime="application/json",
                    use_container_width=True,
                )
                ex2.download_button(
                    "Export report",
                    data=report_payload,
                    file_name="workspace_report.md",
                    mime="text/markdown",
                    use_container_width=True,
                )

            with ws_history:
                history_text = build_history_view_text(st.session_state.chat_history, max_messages=30)
                if history_text:
                    history_selection = render_selectable_text(
                        history_text,
                        key="history_view",
                        height=420,
                        button_label="Use selection",
                    )
                    maybe_add_selection(
                        history_selection,
                        "last_selection_history",
                        label="History",
                        page=None,
                        source="history",
                    )
                else:
                    st.info("No chat history yet.")

    with col_chat:
        draft_context = (st.session_state.composer_text or "").strip()
        if draft_context:
            st.caption("Selected context queued for next message")
            preview = draft_context.replace("\n", " ")
            if len(preview) > 180:
                preview = preview[:177] + "..."
            st.code(preview, language="text")
            if st.button("Clear selected context", key="clear_draft_ctx", use_container_width=False):
                st.session_state.composer_text = ""
                st.rerun()

        user_question = st.chat_input("Message", key="main_chat_input")

    if user_question:
        draft_context = (st.session_state.composer_text or "").strip()
        if draft_context:
            if draft_context not in user_question:
                pending_question = f"{draft_context}\n\n{user_question}"
            else:
                pending_question = user_question
            st.session_state.composer_text = ""
        else:
            pending_question = user_question

        if is_code_prompt(pending_question):
            st.session_state.workspace_visible = True

    if pending_question:
        handle_question(pending_question)
