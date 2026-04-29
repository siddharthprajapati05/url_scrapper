"""
LexaRAG — AI-Powered News Research Assistant
Main Streamlit application with premium dark UI, chat history,
model selection, and industry-grade architecture.
"""

import time
import logging
import streamlit as st

from config import config, AVAILABLE_MODELS
from core.document_processor import DocumentProcessor
from core.llm_manager import LLMManager
from core.vector_store import VectorStoreManager

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# --- Page Configuration ---
st.set_page_config(
    page_title="LexaRAG — AI News Research",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ╔══════════════════════════════════════════════════════════════╗
# ║                    CUSTOM CSS THEME                         ║
# ╚══════════════════════════════════════════════════════════════╝
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

:root {
    --bg-primary: #0d0b1a;
    --bg-secondary: #151225;
    --bg-card: #1c1835;
    --bg-card-hover: #241f42;
    --accent: #7c6aef;
    --accent-light: #a78bfa;
    --accent-glow: rgba(124, 106, 239, 0.18);
    --text-primary: #e8ecf4;
    --text-secondary: #8994a8;
    --text-muted: #5a5478;
    --border: #2a2545;
    --success: #34d399;
    --warning: #fbbf24;
    --error: #f87171;
    --radius: 12px;
}

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

.stApp { background: var(--bg-primary); }

/* === Sidebar === */
[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { font-family: 'Inter', sans-serif !important; }

/* === Inputs === */
.stTextInput > div > div > input,
.stSelectbox > div > div > div {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--text-primary) !important;
    padding: 12px 16px !important;
    font-size: 14px !important;
    transition: border-color 0.25s, box-shadow 0.25s !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-glow) !important;
}
.stTextInput > div > div > input::placeholder { color: var(--text-muted) !important; }

/* === Buttons === */
.stButton > button {
    background: linear-gradient(135deg, #6c63ff 0%, #a78bfa 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: var(--radius) !important;
    padding: 11px 28px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    letter-spacing: 0.4px !important;
    transition: transform 0.2s, box-shadow 0.2s !important;
    box-shadow: 0 4px 18px rgba(108,99,255,0.30) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(108,99,255,0.45) !important;
}

/* === Headers === */
h1, h2, h3 { color: var(--text-primary) !important; font-family: 'Inter', sans-serif !important; }

/* === Alerts === */
[data-testid="stAlert"] {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}

/* === Expander === */
.streamlit-expanderHeader {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--text-primary) !important;
}

/* === Scrollbar === */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

/* === Custom Elements === */
.hero-title {
    font-size: 2.6rem; font-weight: 800; line-height: 1.15;
    background: linear-gradient(135deg, #7c6aef, #a78bfa, #c084fc, #e9d5ff);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin-bottom: 0;
}
.hero-sub {
    font-size: 1.05rem; color: var(--text-secondary);
    margin-top: 8px; font-weight: 400; line-height: 1.6;
}

.card {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 14px; padding: 22px; min-height: 160px;
    transition: border-color 0.3s, box-shadow 0.3s, transform 0.3s;
}
.card:hover {
    border-color: var(--accent); box-shadow: 0 0 24px var(--accent-glow);
    transform: translateY(-3px);
}
.card-icon { font-size: 1.5rem; margin-bottom: 12px; }
.card-title { font-size: 0.92rem; font-weight: 600; color: var(--text-primary); margin-bottom: 6px; }
.card-desc { font-size: 0.78rem; color: var(--text-secondary); line-height: 1.55; }

.empty-state {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 16px; padding: 60px 40px; text-align: center;
    margin-top: 16px;
}
.empty-sparkle {
    font-size: 1.6rem; color: var(--text-muted); margin-bottom: 16px;
}
.empty-text {
    font-size: 0.9rem; color: var(--text-muted); line-height: 1.6;
}

.stat-box {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 12px; padding: 18px; text-align: center;
}
.stat-val {
    font-size: 1.6rem; font-weight: 700;
    background: linear-gradient(135deg, #7c6aef, #a78bfa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}
.stat-lbl {
    font-size: 0.7rem; color: var(--text-muted);
    text-transform: uppercase; letter-spacing: 1px; margin-top: 2px;
}

.chat-msg {
    padding: 20px 24px; border-radius: 14px; margin-bottom: 12px;
    font-size: 0.92rem; line-height: 1.75; word-wrap: break-word;
}
.chat-user {
    background: linear-gradient(135deg, rgba(124,106,239,0.12), rgba(167,139,250,0.08));
    border: 1px solid rgba(124,106,239,0.25); color: var(--text-primary);
}
.chat-bot {
    background: var(--bg-card); border: 1px solid var(--border); color: var(--text-primary);
}
.chat-label {
    font-size: 0.72rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 1px; margin-bottom: 8px;
}
.chat-label-user { color: var(--accent-light); }
.chat-label-bot { color: var(--success); }

.source-chip {
    display: inline-flex; align-items: center; gap: 5px;
    background: rgba(124,106,239,0.10); border: 1px solid rgba(124,106,239,0.22);
    border-radius: 8px; padding: 6px 12px; margin: 3px;
    font-size: 0.78rem; color: var(--accent-light);
    transition: background 0.2s; word-break: break-all;
}
.source-chip:hover { background: rgba(124,106,239,0.22); }
.source-chip-pdf {
    background: rgba(52,211,153,0.10); border-color: rgba(52,211,153,0.28);
    color: #34d399;
}
.source-chip-pdf:hover { background: rgba(52,211,153,0.22); }

/* File uploader */
[data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border: 1px dashed var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 8px !important;
}

.sidebar-box {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 12px; padding: 18px; margin-bottom: 14px;
}
.sidebar-label {
    font-size: 0.7rem; font-weight: 600; color: var(--text-muted);
    text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 8px;
}

.gradient-bar {
    height: 3px; border-radius: 2px; margin: 16px 0;
    background: linear-gradient(90deg, var(--accent), var(--accent-light), #c084fc);
}

.processing-step {
    display: flex; align-items: center; gap: 10px;
    padding: 8px 0; color: var(--text-secondary); font-size: 0.88rem;
}

.metric-time {
    font-size: 0.75rem; color: var(--text-muted); margin-top: 8px;
    font-style: italic;
}

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(16px); }
    to { opacity: 1; transform: translateY(0); }
}
.fade-up { animation: fadeUp 0.5s ease-out; }

@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.5} }
.pulse { animation: pulse 1.8s infinite; }
</style>
""", unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════╗
# ║                   SESSION STATE INIT                        ║
# ╚══════════════════════════════════════════════════════════════╝
defaults = {
    "retriever": None,
    "vectorstore": None,
    "metadata": {},
    "chat_history": [],           # list of {"role": "user"|"assistant", "content": ..., "sources": [...], "time": ...}
    "processing": False,
    "selected_model_idx": 0,
    "pdf_sources": set(),         # filenames that came from PDFs (for citation styling)
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ╔══════════════════════════════════════════════════════════════╗
# ║                       SIDEBAR                              ║
# ╚══════════════════════════════════════════════════════════════╝
with st.sidebar:
    # Brand
    st.markdown("""
    <div style="text-align:center; padding: 12px 0 4px;">
        <div style="font-size:2.4rem;">🧠</div>
        <div style="font-size:1.25rem; font-weight:700;
             background:linear-gradient(135deg,#7c6aef,#a78bfa,#c084fc);
             -webkit-background-clip:text;-webkit-text-fill-color:transparent;
             background-clip:text; margin-top:2px;">LexaRAG</div>
        <div style="font-size:0.72rem;color:#8994a8;margin-top:2px;">
            AI Research Assistant
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="gradient-bar"></div>', unsafe_allow_html=True)


    # --- URL Inputs ---
    st.markdown('<div class="sidebar-label">NEWS ARTICLE URLS</div>', unsafe_allow_html=True)

    urls = []
    for i in range(config.max_urls):
        url = st.text_input(
            f"URL {i+1}",
            placeholder=f"https://example.com/article-{i+1}",
            label_visibility="collapsed",
            key=f"url_{i}",
        )
        urls.append(url)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # --- PDF Uploader ---
    st.markdown('<div class="sidebar-label">UPLOAD PDFS</div>', unsafe_allow_html=True)
    uploaded_pdfs = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key="pdf_uploader",
    )
    if uploaded_pdfs and len(uploaded_pdfs) > 3:
        st.warning("⚠️ Max 3 PDFs allowed. Only the first 3 will be processed.")
        uploaded_pdfs = uploaded_pdfs[:3]

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        process_clicked = st.button("🚀 Process", use_container_width=True)
    with col_btn2:
        clear_clicked = st.button("🗑️ Clear", use_container_width=True)

    if clear_clicked:
        for key in defaults:
            st.session_state[key] = defaults[key]
        st.rerun()

    st.markdown('<div class="gradient-bar"></div>', unsafe_allow_html=True)

    # Powered by
    active_model = AVAILABLE_MODELS[st.session_state.selected_model_idx]
    st.markdown(f"""
    <div style="text-align:center;padding-top:4px;">
        <div style="font-size:0.68rem;color:#4b5563;">
            Powered by <b style="color:#a78bfa;">{active_model.name}</b> · FAISS · HuggingFace
        </div>
    </div>
    """, unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════╗
# ║                    MAIN CONTENT                             ║
# ╚══════════════════════════════════════════════════════════════╝

# --- Validate config ---
config_errors = config.validate()
if config_errors:
    for err in config_errors:
        st.error(f"⚠️ Configuration Error: {err}")
    st.stop()

# --- Initialize managers ---
doc_processor = DocumentProcessor(config)
llm_manager = LLMManager(config)
vs_manager = VectorStoreManager(config)

# --- Hero ---
st.markdown("""
<div class="fade-up">
    <div class="hero-title">AI-Powered Research Assistant</div>
    <div class="hero-sub">
        Paste URLs or upload PDFs — get cited answers via semantic search and Gemini
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)


# --- Feature Cards (shown when idle) ---
if not st.session_state.retriever and not process_clicked:
    cols = st.columns(4)
    features = [
        ("🔗", "Multi-URL ingestion", "Analyze up to 3 news articles simultaneously"),
        ("📄", "PDF support", "Upload PDFs alongside URLs, indexed together"),
        ("⚡", "Gemini AI", "Fast, accurate answers with source citations"),
        ("💬", "Chat history", "Full memory for natural follow-up questions"),
    ]
    for col, (icon, title, desc) in zip(cols, features):
        with col:
            st.markdown(f"""
            <div class="card fade-up">
                <div class="card-icon">{icon}</div>
                <div class="card-title">{title}</div>
                <div class="card-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div class="empty-state fade-up">
        <div class="empty-sparkle">✦</div>
        <div class="empty-text">
            Paste URLs and/or upload PDFs in the sidebar to get started
        </div>
    </div>
    """, unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════╗
# ║                   URL PROCESSING                            ║
# ╚══════════════════════════════════════════════════════════════╝
status_area = st.empty()

if process_clicked:
    raw_urls = [u for u in urls if u.strip()]
    has_urls = bool(raw_urls)
    has_pdfs = bool(uploaded_pdfs)

    if not has_urls and not has_pdfs:
        st.sidebar.error("⚠️ Provide at least one URL or upload a PDF.")
    else:
        # Warn about invalid URLs (don't abort if PDFs are present)
        if has_urls:
            invalid = [u for u in raw_urls if not DocumentProcessor.validate_url(u)]
            if invalid:
                for u in invalid:
                    st.sidebar.warning(f"⚠️ Invalid URL skipped: {u[:50]}...")

        valid_urls = DocumentProcessor.clean_urls(raw_urls) if has_urls else []

        try:
            all_docs = []
            all_errors = []
            url_count = 0
            pdf_count = 0
            pdf_filenames = set()

            # --- Load URL documents ---
            if valid_urls:
                status_area.markdown("""
                <div class="processing-step fade-up">
                    <span style="font-size:1.2rem">📥</span>
                    <span>Loading articles from URLs...</span>
                </div>
                """, unsafe_allow_html=True)
                url_docs, url_errors = doc_processor.load_documents(valid_urls)
                all_docs.extend(url_docs)
                all_errors.extend(url_errors)
                url_count = len(valid_urls)

            # --- Load PDF documents ---
            if has_pdfs:
                status_area.markdown("""
                <div class="processing-step fade-up">
                    <span style="font-size:1.2rem">📄</span>
                    <span>Loading PDF files...</span>
                </div>
                """, unsafe_allow_html=True)
                pdf_docs, pdf_errors = doc_processor.load_pdfs(uploaded_pdfs)
                all_docs.extend(pdf_docs)
                all_errors.extend(pdf_errors)
                pdf_count = len(uploaded_pdfs)
                pdf_filenames = {f.name for f in uploaded_pdfs}

            # --- Show any errors ---
            if all_errors:
                for err in all_errors:
                    st.error(f"❌ {err}")

            if not all_docs:
                st.warning("⚠️ No content was extracted. Check your URLs and PDFs.")
                status_area.empty()
            else:
                # --- Build vector store ---
                status_area.markdown("""
                <div class="processing-step fade-up">
                    <span style="font-size:1.2rem">🧬</span>
                    <span>Building vector store...</span>
                </div>
                """, unsafe_allow_html=True)

                chunks = doc_processor.split_documents(all_docs)
                vectorstore = vs_manager.create_store(chunks)
                retriever = vs_manager.get_retriever(vectorstore)

                metadata = {
                    "total_urls": url_count,
                    "total_pdfs": pdf_count,
                    "total_chunks": len(chunks),
                    "avg_chunk_size": sum(len(c.page_content) for c in chunks) // max(len(chunks), 1),
                    "sources": list(set(doc.metadata.get("source", "Unknown") for doc in all_docs)),
                }

                # Save to session
                st.session_state.retriever = retriever
                st.session_state.vectorstore = vectorstore
                st.session_state.metadata = metadata
                st.session_state.pdf_sources = pdf_filenames
                st.session_state.chat_history = []
                status_area.empty()

                # Build a friendly summary line
                parts = []
                if url_count:
                    parts.append(f"{url_count} URL{'s' if url_count > 1 else ''}")
                if pdf_count:
                    parts.append(f"{pdf_count} PDF{'s' if pdf_count > 1 else ''}")
                summary = " + ".join(parts)

                st.markdown(f"""
                <div style="background:var(--bg-card);border:1px solid var(--success);
                     border-radius:14px;padding:20px;text-align:center;" class="fade-up">
                    <div style="font-size:1.8rem;margin-bottom:6px;">✅</div>
                    <div style="font-size:1rem;font-weight:600;color:var(--success);">
                        {summary} processed — {len(chunks)} chunks indexed
                    </div>
                    <div style="font-size:0.8rem;color:var(--text-muted);margin-top:4px;">
                        Ready for questions
                    </div>
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            status_area.empty()
            logger.exception("Processing failed")
            st.error(f"❌ Processing failed: {e}")


# ╔══════════════════════════════════════════════════════════════╗
# ║                    Q&A INTERFACE                            ║
# ╚══════════════════════════════════════════════════════════════╝
if st.session_state.retriever:

    # --- Stats Bar ---
    meta = st.session_state.metadata
    c1, c2, c3, c4 = st.columns(4)
    stats = [
        (str(meta.get("total_urls", 0)), "URLs"),
        (str(meta.get("total_pdfs", 0)), "PDFs"),
        (str(meta.get("total_chunks", 0)), "Chunks"),
        (str(len(st.session_state.chat_history) // 2), "Questions"),
    ]
    for col, (val, lbl) in zip([c1, c2, c3, c4], stats):
        with col:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-val">{val}</div>
                <div class="stat-lbl">{lbl}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # --- Chat History ---
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            with st.chat_message("user", avatar="🧑"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(msg["content"])

                # Sources — distinguish PDF (📄) from URL (🔗) visually
                pdf_srcs = st.session_state.get("pdf_sources", set())
                valid_sources = [s for s in msg.get("sources", []) if s.get("url") and s["url"] != "Unknown"]
                if valid_sources:
                    st.markdown("---")
                    st.caption("SOURCES")
                    chips_html = ""
                    for s in valid_sources:
                        src = s["url"]
                        is_pdf = src in pdf_srcs
                        chip_class = "source-chip source-chip-pdf" if is_pdf else "source-chip"
                        icon = "📄" if is_pdf else "🔗"
                        chips_html += f'<span class="{chip_class}">{icon} {src}</span>'
                    st.markdown(f'<div style="margin-top:6px">{chips_html}</div>', unsafe_allow_html=True)

                # Response time
                if msg.get("time"):
                    st.caption(f"⏱ Answered in {msg['time']:.1f}s")

    # --- Query Input ---
    query = st.chat_input("Ask anything about your documents...")

    if query:
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": query})

        # Build chain with selected model
        active_model = AVAILABLE_MODELS[st.session_state.selected_model_idx]
        try:
            llm = llm_manager.create_llm(active_model)
            chain = llm_manager.build_chain(llm, st.session_state.retriever)

            t0 = time.perf_counter()
            result = chain.invoke({"input": query})
            elapsed = time.perf_counter() - t0

            sources = llm_manager.format_sources(result)

            # Add bot message
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": result["answer"],
                "sources": sources,
                "time": elapsed,
            })

        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "quota" in error_msg.lower():
                friendly = (
                    "⚠️ API rate limit reached. Please wait a minute and try again, "
                    "or switch to a different model in the sidebar."
                )
            elif "404" in error_msg and "not found" in error_msg.lower():
                friendly = (
                    f"⚠️ Model `{active_model.model_id}` is not available. "
                    "Try selecting a different model from the sidebar."
                )
            else:
                friendly = f"⚠️ Error: {error_msg}"

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": friendly,
                "sources": [],
                "time": None,
            })
            logger.error(f"Query failed: {error_msg}")

        st.rerun()