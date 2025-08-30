"""
Studymate - PDF Q&A Assistant (LOCAL LLM VERSION)
Runs the LLM locally with Hugging Face transformers (GPU preferred, CPU OK but slow).
RAG: PyMuPDF + Sentence-Transformers + FAISS; Local generation: transformers pipeline.
"""

import re
import json
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, cast

import streamlit as st
import numpy as np
import faiss
import fitz  # PyMuPDF
import torch

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ---------------- Config ----------------
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_OVERLAP = 200
DEFAULT_TOP_K = 5
DEFAULT_MODEL_ID = "ibm-granite/granite-3.3-2b-instruct"  # change in sidebar if needed
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

# ---------------- Local LLM (transformers) ----------------
def build_prompt(question: str, context: str) -> str:
    return (
        "You are a helpful academic assistant. Answer ONLY using the provided context.\n"
        "If the context is insufficient, say so clearly.\n\n"
        f"Question: {question}\n\nContext:\n{context}\n"
    )

@st.cache_resource(show_spinner=True)
def load_local_pipeline(
    model_id: str,
    hf_token: Optional[str],
    use_trust_remote_code: bool,
    dtype_choice: str,
):
    """
    Loads tokenizer + model and returns a text-generation pipeline.
    device_map='auto' uses accelerate to place weights on available GPU/CPU.
    """
    # dtype selection
    dtype = None
    if dtype_choice == "float16":
        dtype = torch.float16
    elif dtype_choice == "bfloat16":
        dtype = torch.bfloat16
    elif dtype_choice == "float32":
        dtype = torch.float32  # slowest / most compatible

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
        device_map="auto",  # requires accelerate
        trust_remote_code=use_trust_remote_code,
        low_cpu_mem_usage=True,
    )

    gen = pipeline(
        "text-generation",
        model=mdl,
        tokenizer=tok,
        # leave device=None so pipeline honors device_map on the model
        # framework is torch
    )
    return gen, tok

def call_local_llm(
    question: str,
    context: str,
    pipe,
    tok,
    max_new_tokens: int = 512,
    temperature: float = 0.2,
    top_p: float = 0.95,
) -> str:
    prompt = build_prompt(question, context[:MAX_CONTEXT_CHARS])

    # Ensure prompt length fits; truncate if needed
    inputs = tok(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=tok.model_max_length - max_new_tokens - 8,
    )
    # Send tensors to the same device as the model’s first weight shard
    if hasattr(pipe.model, "hf_device_map"):
        first_device = next(iter(set(pipe.model.hf_device_map.values())))
    else:
        first_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    inputs = {k: v.to(pipe.model.device if hasattr(pipe.model, "device") else first_device) for k, v in inputs.items()}

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
    # Return only the completion after the prompt
    return text[len(tok.decode(inputs["input_ids"][0], skip_special_tokens=True)) :].strip()

# ---------------- Streamlit wiring ----------------
def init_session_state():
    if "chunks" not in st.session_state:
        st.session_state["chunks"] = []
    if "index" not in st.session_state:
        st.session_state["index"] = None
    if "qa_history" not in st.session_state:
        st.session_state["qa_history"] = []
    if "embeddings" not in st.session_state:
        st.session_state["embeddings"] = None

def main():
    st.set_page_config(page_title="Studymate (Local)", page_icon="📘", layout="wide")
    init_session_state()

    with st.sidebar:
        st.header("⚙️ Local LLM Settings")
        model_id = st.text_input(
            "HF Model ID",
            value=DEFAULT_MODEL_ID,
            help="e.g., ibm-granite/granite-3.3-2b-instruct or ibm-granite/granite-3.0-2b-instruct",
        )
        hf_token = st.text_input(
            "Hugging Face Token (read)",
            type="password",
            help="Required to download model weights the first time.",
        )
        trust_remote = st.checkbox("trust_remote_code (only if model readme says so)", value=False)
        dtype_choice = st.selectbox("dtype", ["bfloat16", "float16", "float32"], index=0)
        max_new_tokens = st.slider("Max new tokens", 64, 1024, 512, 64)
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
            st.session_state["qa_history"] = []
            st.experimental_rerun()

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
    st.header("💬 Ask a question")

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

    c1, c2 = st.columns([5, 1])
    with c1:
        question = st.text_input("Your question:", placeholder="e.g., Summarize section 2’s main idea")
    with c2:
        ask = st.button("🔍 Ask", type="primary", disabled=not bool(question))

    if ask and question:
        with st.spinner("Retrieving relevant chunks…"):
            emb_model = load_embedding_model()
            retrieved = query_index(question, st.session_state["index"], st.session_state["chunks"], emb_model, top_k)
        if not retrieved:
            st.error("No relevant content found.")
            return

        context = format_context(retrieved)
        with st.spinner("Generating locally…"):
            try:
                answer = call_local_llm(question, context, gen, tok, max_new_tokens, temperature, top_p)
            except Exception as e:
                st.error(f"Local generation failed: {e}")
                return

        st.subheader("📖 Answer")
        st.markdown(answer)

        qa_hist = cast(List[dict], st.session_state["qa_history"])
        qa_hist.append({"question": question, "answer": answer, "chunks": retrieved})

        st.subheader("📚 References")
        for i, (chunk, dist) in enumerate(retrieved, 1):
            score = 1 / (1 + dist)
            with st.expander(f"Ref {i} — {chunk.source_file} (p.{chunk.page_num}) • relevance≈{score:.3f}"):
                txt = chunk.text
                st.text(txt[:1000] + ("..." if len(txt) > 1000 else ""))

if __name__ == "__main__":
    main()
