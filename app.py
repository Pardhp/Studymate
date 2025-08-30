"""
Studymate - PDF Q&A Assistant (LOCAL LLM VERSION)
Runs the LLM locally with Hugging Face transformers (GPU preferred, CPU OK but slow).
RAG: PyMuPDF + Sentence-Transformers + FAISS; Local generation: transformers pipeline.
"""

import re
import json
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import streamlit as st
import numpy as np
import faiss
import fitz  # PyMuPDF
import torch

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

import uuid
from datetime import datetime

# ---- Chat state ----
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "You are StudyMate, a helpful academic assistant."}]

if "qa_log" not in st.session_state:
    st.session_state.qa_log = []

if "run_id" not in st.session_state:
    st.session_state.run_id = uuid.uuid4().hex[:8]

# ---------------- Config ----------------
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_OVERLAP = 200
DEFAULT_TOP_K = 5
DEFAULT_MODEL_ID = "ibm-granite/granite-3.3-2b-instruct"
MAX_CONTEXT_CHARS = 4000  # keep prompt reasonable

# ---------------- Data ----------------
@dataclass
class Chunk:
    text: str
    page_num: int
    chunk_id: int
    source_file: str

# ---------------- PDF -> Chunks ----------------
def extract_text_from_pdf(pdf_file) -> List[Dict]:
    pages: List[Dict] = []
    try:
        pdf_bytes = pdf_file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text()
            if text and text.strip():
                pages.append({"page": page_num, "text": text.strip()})
        doc.close()
    except Exception as e:
        st.error(f"Error reading PDF {getattr(pdf_file, 'name', 'file')}: {e}")
    return pages

def chunk_text(
    pages: List[Dict],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    filename: str = "",
) -> List[Chunk]:
    chunks: List[Chunk] = []
    cid = 0
    for page in pages:
        page_num = page["page"]
        text = re.sub(r"\s+", " ", page["text"]).strip()
        if not text:
            continue
        start, n = 0, len(text)
        while start < n:
            end = min(start + chunk_size, n)
            if end < n:
                last_period = text.rfind(".", start, end)
                if last_period > start + chunk_size // 2:
                    end = last_period + 1
            seg = text[start:end].strip()
            if seg:
                chunks.append(Chunk(seg, page_num, cid, filename))
                cid += 1
            start = end if end == n else max(end - overlap, start + 1)
    return chunks

# ---------------- Embeddings & Retrieval ----------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def create_embeddings(chunks: List[Chunk], model) -> np.ndarray:
    texts = [c.text for c in chunks]
    emb = model.encode(texts, show_progress_bar=False)
    return np.array(emb, dtype="float32")

def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def query_index(
    question: str,
    index: faiss.IndexFlatL2,
    chunks: List[Chunk],
    model,
    top_k: int = DEFAULT_TOP_K,
) -> List[Tuple[Chunk, float]]:
    if not chunks:
        return []
    qvec = model.encode([question]).astype("float32")
    k = min(top_k, len(chunks))
    distances, indices = index.search(qvec, k)
    out: List[Tuple[Chunk, float]] = []
    for idx, dist in zip(indices[0], distances[0]):
        if 0 <= idx < len(chunks):
            out.append((chunks[idx], float(dist)))
    return out

def format_context(retrieved: List[Tuple[Chunk, float]], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    parts = []
    total = 0
    for chunk, _ in retrieved:
        piece = f"[Page {chunk.page_num} - {chunk.source_file}]\n{chunk.text}\n"
        if total + len(piece) > max_chars:
            break
        parts.append(piece)
        total += len(piece)
    return "\n".join(parts)

# ---------- Chat helpers ---------- 
def build_history_text(k: int = 6) -> str:
    msgs = st.session_state.messages[1:]  # skip system
    if not msgs:
        return ""
    tail = msgs[-k:] if len(msgs) > k else msgs
    lines = []
    for m in tail:
        role = "User" if m["role"] == "user" else "Assistant"
        lines.append(f"{role}: {m['content']}")
    return "\n".join(lines)

def with_context(user_text: str, retrieved_chunks: List[Tuple[Chunk, float]] | None) -> str:
    if not retrieved_chunks:
        return user_text
    parts = []
    total = 0
    for (chunk, _) in retrieved_chunks:
        piece = f"[p.{chunk.page_num} – {chunk.source_file}] {chunk.text}"
        if total + len(piece) > MAX_CONTEXT_CHARS:
            break
        parts.append(piece)
        total += len(piece)
    ctx = "\n".join(parts)
    return f"{user_text}\n\n---\nUse these sources when answering:\n{ctx}"

# ---------------- Local LLM ----------------
def build_prompt(question: str, context: str, history_text: str = "") -> str:
    preface = (
        "You are a helpful academic assistant. Use only the provided context and, if helpful, "
        "the prior conversation to resolve follow-up references. If the context is insufficient, say so clearly.\n"
    )
    hist = f"\nConversation so far:\n{history_text}\n" if history_text.strip() else ""
    return f"{preface}{hist}\nQuestion: {question}\n\nContext:\n{context}\n"

@st.cache_resource(show_spinner=True)
def load_local_pipeline(
    model_id: str,
    hf_token: Optional[str],
    use_trust_remote_code: bool,
    dtype_choice: str,
):
    dtype = None
    if dtype_choice == "float16":
        dtype = torch.float16
    elif dtype_choice == "bfloat16":
        dtype = torch.bfloat16
    elif dtype_choice == "float32":
        dtype = torch.float32

    tok = AutoTokenizer.from_pretrained(
        model_id,
        token=hf_token,
        use_fast=True,
        trust_remote_code=use_trust_remote_code,
    )
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=hf_token,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=use_trust_remote_code,
        low_cpu_mem_usage=True,
    )

    gen = pipeline("text-generation", model=mdl, tokenizer=tok)
    return gen, tok

def call_local_llm_with_history(
    question: str,
    context: str,
    pipe,
    tok,
    history_text: str = "",
    max_new_tokens: int = 512,
    temperature: float = 0.2,
    top_p: float = 0.95,
) -> str:
    prompt = build_prompt(question, context[:MAX_CONTEXT_CHARS], history_text)
    inputs = tok(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=tok.model_max_length - max_new_tokens - 8,
    )
    device = pipe.model.device if hasattr(pipe.model, "device") else "cuda" if torch.cuda.is_available() else "cpu"
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out = pipe.model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True if temperature > 0 else False,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
        repetition_penalty=1.05,
    )
    text = tok.decode(out[0], skip_special_tokens=True)
    return text[len(tok.decode(inputs["input_ids"][0], skip_special_tokens=True)):].strip()

# ---------------- Streamlit wiring ----------------
def init_session_state():
    if "chunks" not in st.session_state:
        st.session_state["chunks"] = []
    if "index" not in st.session_state:
        st.session_state["index"] = None
    if "embeddings" not in st.session_state:
        st.session_state["embeddings"] = None

def local_css(file_name: str):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="Studymate (Local)", page_icon="📘", layout="wide")
    init_session_state()
    local_css("styles.css")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Local LLM Settings")
        model_id = st.text_input("HF Model ID", value=DEFAULT_MODEL_ID)
        hf_token = st.text_input("Hugging Face Token (read)", type="password")
        trust_remote = st.checkbox("trust_remote_code", value=False)
        dtype_choice = st.selectbox("dtype", ["bfloat16", "float16", "float32"], index=0)
        max_new_tokens = st.slider("Max new tokens", 64, 512, 64, 64)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
        top_p = st.slider("Top-p", 0.1, 1.0, 0.95, 0.05)

        st.divider()
        st.header("📊 Retrieval Settings")
        chunk_size = st.slider("Chunk Size (chars)", 500, 2000, DEFAULT_CHUNK_SIZE, 100)
        overlap = st.slider("Chunk Overlap (chars)", 50, 500, DEFAULT_OVERLAP, 50)
        top_k = st.slider("Top-K Chunks", 1, 10, DEFAULT_TOP_K)

        if st.button("🔄 Reset Session"):
            st.session_state["index"] = None
            st.session_state["chunks"] = []
            st.session_state["embeddings"] = None
            st.session_state.messages = st.session_state.messages[:1]
            st.session_state.qa_log = []
            st.experimental_rerun()

    # Main UI
    st.title("📘 Studymate (Local Transformers)")
    st.caption("Ask your PDFs anything — RAG over local PDFs + local Granite (or other HF model)")

    # Upload + process
    st.header("📄 Upload PDFs")
    files = st.file_uploader("Choose PDFs", type=["pdf"], accept_multiple_files=True)
    if files and st.button("📥 Process PDFs", type="primary"):
        with st.spinner("Extracting & chunking…"):
            all_chunks: List[Chunk] = []
            prog = st.progress(0.0)
            for i, f in enumerate(files):
                pages = extract_text_from_pdf(f)
                if pages:
                    all_chunks.extend(chunk_text(pages, chunk_size, overlap, f.name))
                prog.progress((i + 1) / len(files))
            st.session_state["chunks"] = all_chunks

        if st.session_state["chunks"]:
            with st.spinner("Embedding & indexing…"):
                emb_model = load_embedding_model()
                E = create_embeddings(st.session_state["chunks"], emb_model)
                st.session_state["embeddings"] = E
                st.session_state["index"] = build_faiss_index(E)
            st.success(f"✅ {len(files)} PDF(s) → {len(st.session_state['chunks'])} chunks")
        else:
            st.error("No extractable text found.")

    st.divider()
    st.header("💬 Chat")

    if not st.session_state["chunks"] or st.session_state["index"] is None:
        st.info("Upload and process PDFs first.")
        return

    # Load local model pipeline once
    try:
        with st.spinner("Loading local model (first time may take a while)…"):
            gen, tok = load_local_pipeline(model_id, hf_token or None, trust_remote, dtype_choice)
    except Exception as e:
        st.error(f"Failed to load local model: {e}")
        return

    left, right = st.columns([2, 1], gap="large")

    with left:
        for m in st.session_state.messages[1:]:
            who = "user" if m["role"] == "user" else "assistant"
            st.chat_message(who).markdown(m["content"])

        user_text = st.chat_input("Ask a question or a follow-up…")
        if user_text:
            with st.spinner("Retrieving relevant chunks…"):
                emb_model = load_embedding_model()
                retrieved = query_index(user_text, st.session_state["index"], st.session_state["chunks"], emb_model, top_k)

            st.session_state.messages.append({"role": "user", "content": user_text})
            st.chat_message("user").markdown(user_text)

            if not retrieved:
                st.chat_message("assistant").markdown("_I couldn't find relevant passages in the uploaded PDFs._")
            else:
                history_text = build_history_text(k=6)
                with st.spinner("Generating locally…"):
                    try:
                        answer = call_local_llm_with_history(
                            question=user_text,
                            context=format_context(retrieved),
                            pipe=gen,
                            tok=tok,
                            history_text=history_text,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            top_p=top_p,
                        )
                    except Exception as e:
                        answer = f"_Local generation failed: {e}_"

                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.chat_message("assistant").markdown(answer)

                st.session_state.qa_log.append({
                    "id": uuid.uuid4().hex[:6],
                    "question": user_text,
                    "answer": answer,
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "sources": [f"{c.source_file} (p.{c.page_num})" for (c, _) in retrieved]
                })

                st.subheader("📚 References")
                for i, (chunk, dist) in enumerate(retrieved, 1):
                    score = 1 / (1 + dist)
                    with st.expander(f"Ref {i} — {chunk.source_file} (p.{chunk.page_num}) • relevance≈{score:.3f}"):
                        txt = chunk.text
                        st.text(txt[:1000] + ("..." if len(txt) > 1000 else ""))

    with right:
        st.subheader("Q&A History")
        if not st.session_state.qa_log:
            st.info("No questions yet. Your full conversation will appear here.")
        else:
            for row in reversed(st.session_state.qa_log):
                with st.expander(f'[{row["time"]}] Q: {row["question"][:60]}', expanded=False):
                    st.markdown(f'**Q:** {row["question"]}')
                    st.markdown(f'**A:** {row["answer"]}')
                    if row["sources"]:
                        st.caption("Sources:")
                        for s in row["sources"]:
                            st.write(f"- {s}")

            json_bytes = json.dumps(st.session_state.qa_log, ensure_ascii=False, indent=2).encode("utf-8")
            st.download_button(
                "Download Q&A (JSON)",
                data=json_bytes,
                file_name=f"studymate_log_{st.session_state.run_id}.json",
                mime="application/json",
            )

            txt_blob = "\n\n".join(
                [f'[{r["time"]}] Q: {r["question"]}\nA: {r["answer"]}' for r in st.session_state.qa_log]
            ).encode("utf-8")
            st.download_button(
                "Download Q&A (TXT)",
                data=txt_blob,
                file_name=f"studymate_log_{st.session_state.run_id}.txt",
                mime="text/plain",
            )

        st.divider()
        if st.button("Clear Current Chat"):
            st.session_state.messages = st.session_state.messages[:1]
            st.toast("Chat cleared.")
        if st.button("Clear Q&A History"):
            st.session_state.qa_log = []
            st.toast("History cleared.")


if __name__ == "__main__":
    main()
