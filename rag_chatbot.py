import os
import re
import time
import pickle
import numpy as np
import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer

# ---------- Config ----------
st.set_page_config(page_title="RAG Chatbot (FAQ TontonUp)", page_icon="🤖", layout="centered")

USE_GEMINI = True
MODEL_NAME = "models/gemini-1.5-flash"  
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

try:
    import google.generativeai as genai
except Exception:
    USE_GEMINI = False
    genai = None

# ---------- Cached loaders ----------
@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

@st.cache_resource(show_spinner=False)
def load_index_and_data():
    index = faiss.read_index("faq_index.faiss")
    with open("faq_data.pkl", "rb") as f:
        faq_data = pickle.load(f)
    return index, faq_data

embedding_model = load_embedding_model()
index, faq_data = load_index_and_data()

# Integrity check (catch stale/misaligned files early)
try:
    assert index.ntotal == len(faq_data), f"Index/data mismatch: index={index.ntotal}, data={len(faq_data)}"
except AssertionError as e:
    st.error(str(e))
    st.stop()

# ---------- Sidebar ----------
with st.sidebar:
    st.subheader("Tetapan")

    # Debug toggle FIRST
    show_debug = st.checkbox("Tunjuk panel debug", value=False)

    # Only show sliders if debug is ON; else use safe defaults
    if show_debug:
        top_k = st.slider("Dokumen dirujuk (top_k)", 1, 5, 3)
        temperature = st.slider("Kreativiti (temperature)", 0.0, 1.0, 0.2)
        st.markdown("---")
    else:
        top_k = 3
        temperature = 0.2

    # Bigger API label
    st.markdown("<span style='font-size:1.1em; font-weight:600;'>API: Gemini</span>", unsafe_allow_html=True)
    if not GEMINI_API_KEY:
        st.warning("GEMINI_API_KEY tidak ditetapkan. Aplikasi akan guna fallback ekstraktif.")

    # Extra debug info only when ON
    if show_debug:
        st.markdown("---")
        st.write(f"📦 Working dir: `{os.getcwd()}`")
        st.write(f"🗂️ FAQ docs loaded: **{len(faq_data)}**")
        st.write(f"🔎 Index size: **{index.ntotal}**")
        if os.path.exists("faq_data.pkl"):
            st.write("📅 faq_data.pkl mtime:", time.ctime(os.path.getmtime("faq_data.pkl")))
        if os.path.exists("faq_index.faiss"):
            st.write("📅 faq_index.faiss mtime:", time.ctime(os.path.getmtime("faq_index.faiss")))

# ---------- Gemini config ----------
model = None
if USE_GEMINI and GEMINI_API_KEY and genai is not None:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(MODEL_NAME)
    except Exception as e:
        st.toast(f"Gagal memuatkan Gemini: {e}")

# ---------- Retrieval ----------
def retrieve_chunks(user_question: str, top_k: int = 3):
    """Return reranked chunks without dropping any; prefer Q+A text for recall."""
    k = min(int(top_k), index.ntotal)
    #k = max(3, min(int(top_k), index.ntotal))
    if k <= 0:
        return [], []

    # Semantic search
    query_emb = embedding_model.encode([user_question])
    D, I = index.search(np.array(query_emb), k)

    # Build the same style you EMBEDDED (ideally Q+A). If your index is answer-only, this still works.
    base = []
    for idx in I[0]:
        qa = faq_data[idx]
        if isinstance(qa, dict) and "question" in qa and "answer" in qa:
            text = f"Question: {qa['question']}\nAnswer: {qa['answer']}"
        else:
            # fallback: answer-only
            text = qa["answer"] if isinstance(qa, dict) and "answer" in qa else str(qa)
        base.append(text)

    # Soft keyword bump (re-rank) — DO NOT DROP
    q_words = set(re.findall(r"\w+", user_question.lower()))
    scored = []
    for t in base:
        t_words = set(re.findall(r"\w+", t.lower()))
        scored.append((len(q_words & t_words), t))
    scored.sort(key=lambda x: x[0], reverse=True)

    reranked = [t for _, t in scored]
    return reranked, base

# ---------- Generation ----------
def generate_answer(user_question: str, retrieved_chunks, temperature=0.2, max_retries=2):
    # Strip the "Question:" header so Gemini only sees answers
    def strip_question_prefix(text: str) -> str:
        return re.sub(r"(?is)^question:\s.*?answer:\s*", "", text, count=1).strip()

    # Light pre-format: add newlines before 1), 1., •, -
    def prebreak_lists(s: str) -> str:
        s = re.sub(r"\s+(\d+[\.\)])\s+", r"\n\1 ", s)   # 1. / 1)
        s = re.sub(r"\s+([•\-–])\s+", r"\n\1 ", s)      # bullets
        return s

    chunks = [prebreak_lists(strip_question_prefix(c))[:1200] for c in retrieved_chunks]
    retrieved_context = "\n\n".join(chunks) if chunks else ""

    prompt = (
        "Sila gunakan maklumat di bawah sahaja untuk menjawab soalan pengguna. "
        "JANGAN mereka-reka maklumat. Jika tiada jawapan, balas 'Maklumat tidak tersedia'.\n\n"
        "Pastikan jawapan anda mengandungi semua butiran penting seperti nombor langkah atau bilangan episod.\n"
        "Keperluan pemformatan:\n"
        "• Jika jawapan berbentuk langkah/point, FORMATKAN sebagai senarai bernombor (1., 2., 3.)\n"
        "• Satu langkah setiap baris, ringkas dan jelas.\n"
        "• Jika ada prasyarat/nota, letak di bahagian 'Nota:' di bawah senarai.\n"
        "Bahasa: Bahasa Melayu.\n\n"
        f"MAKLUMAT:\n{retrieved_context}\n\n"
        f"Soalan: {user_question}\n"
        "Jawapan (dalam format Markdown):"
    )

    # Fallback (no LLM)
    if model is None:
        return chunks[0] if chunks else "Maklumat tidak tersedia."

    # Retry + extractive fallback
    for attempt in range(max_retries + 1):
        try:
            response = model.generate_content(
                prompt,
                generation_config={"temperature": float(temperature)}
            )
            txt = (response.text or "").strip()
            if txt:
                # If LLM refuses even though we have context, fall back to extractive
                if "Maklumat tidak tersedia" in txt and chunks:
                    return chunks[0]
                return txt
            # empty -> extractive
            return chunks[0] if chunks else "Maklumat tidak tersedia."
        except Exception as e:
            if ("ResourceExhausted" in str(e) or "429" in str(e)) and attempt < max_retries:
                time.sleep(2 + attempt)
                continue
            return chunks[0] if chunks else "Maklumat tidak tersedia."


# ---------- UI ----------
st.title("🤖 Chatbot FAQ TontonUp (RAG)")

# keep state
if "history" not in st.session_state:
    st.session_state.history = []
if "user_q" not in st.session_state:
    st.session_state.user_q = ""

def on_submit():
    q = st.session_state.user_q.strip()
    if not q:
        return
    #k = min(int(top_k), index.ntotal)  # guard top_k
    k = max(3, min(int(top_k), index.ntotal))
    retrieved, raw_hits = retrieve_chunks(q, top_k=k)
    best_only = retrieved[:1]
    ans = generate_answer(q, best_only, temperature=temperature)
    st.session_state.history.append(
        {"question": q, "answer": ans, "retrieved": best_only, "raw_hits": raw_hits}
    )
    st.session_state.user_q = ""  # clear input

with st.form("ask"):
    st.text_input("Tanya soalan anda di sini …", key="user_q",
                  placeholder="Contoh: Kenapa masih ada iklan selepas melanggan?")
    submitted = st.form_submit_button("Hantar", on_click=on_submit)

# Reset button
if st.button("🗑️ Kosongkan sejarah"):
    st.session_state.history = []

# Render history
for turn in reversed(st.session_state.history):
    st.markdown(f"**Anda:** {turn['question']}")
    st.markdown(f"**Chatbot:**\n\n{turn['answer']}")
    if show_debug:
        with st.expander("🔎 Top-k (mentah) daripada retriever"):
            st.write("\n\n".join(turn["raw_hits"]))
    st.markdown("---")
