import os
import streamlit as st
import pickle
import time
from dotenv import load_dotenv

# --- Modern LangChain Imports ---
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
# CHANGED: Import the Gemini chat model instead of Groq
from langchain_google_genai import ChatGoogleGenerativeAI

# Load API key from .env file
load_dotenv()
# CHANGED: Load the Google API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- Page Configuration ---
st.set_page_config(
    page_title="RockyBot — AI News Research",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Premium Dark UI ---
st.markdown("""
<style>
    /* ====== Google Font ====== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ====== Root Variables ====== */
    :root {
        --bg-primary: #0f1117;
        --bg-secondary: #161b22;
        --bg-card: #1c2333;
        --bg-card-hover: #222b3d;
        --accent-primary: #6c63ff;
        --accent-secondary: #a78bfa;
        --accent-gradient: linear-gradient(135deg, #6c63ff 0%, #a78bfa 50%, #c084fc 100%);
        --text-primary: #e6edf3;
        --text-secondary: #8b949e;
        --text-muted: #484f58;
        --border-color: #30363d;
        --success: #3fb950;
        --warning: #d29922;
        --error: #f85149;
        --glow: 0 0 20px rgba(108, 99, 255, 0.15);
    }

    /* ====== Global Styles ====== */
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif !important;
        color: var(--text-primary);
    }

    .stApp {
        background: var(--bg-primary);
    }

    /* ====== Sidebar ====== */
    [data-testid="stSidebar"] {
        background: var(--bg-secondary) !important;
        border-right: 1px solid var(--border-color) !important;
    }

    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: var(--text-primary) !important;
    }

    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown span,
    [data-testid="stSidebar"] label {
        color: var(--text-secondary) !important;
    }

    /* ====== Input Fields ====== */
    .stTextInput > div > div > input {
        background-color: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 10px !important;
        color: var(--text-primary) !important;
        padding: 12px 16px !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 14px !important;
        transition: all 0.3s ease !important;
    }

    .stTextInput > div > div > input:focus {
        border-color: var(--accent-primary) !important;
        box-shadow: 0 0 0 3px rgba(108, 99, 255, 0.2) !important;
    }

    .stTextInput > div > div > input::placeholder {
        color: var(--text-muted) !important;
    }

    /* ====== Buttons ====== */
    .stButton > button {
        background: var(--accent-gradient) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 10px 24px !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        letter-spacing: 0.3px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(108, 99, 255, 0.3) !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 25px rgba(108, 99, 255, 0.45) !important;
    }

    .stButton > button:active {
        transform: translateY(0px) !important;
    }

    /* ====== Headers ====== */
    h1 {
        color: var(--text-primary) !important;
        font-weight: 800 !important;
    }

    h2, h3 {
        color: var(--text-primary) !important;
        font-weight: 600 !important;
    }

    /* ====== Alert / Info Boxes ====== */
    .stAlert {
        background-color: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 12px !important;
        color: var(--text-secondary) !important;
    }

    [data-testid="stAlert"] {
        background-color: var(--bg-card) !important;
        border-radius: 12px !important;
    }

    /* ====== Spinner ====== */
    .stSpinner > div {
        border-top-color: var(--accent-primary) !important;
    }

    /* ====== Divider ====== */
    hr {
        border-color: var(--border-color) !important;
    }

    /* ====== Scrollbar ====== */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: var(--bg-primary);
    }
    ::-webkit-scrollbar-thumb {
        background: var(--border-color);
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: var(--text-muted);
    }

    /* ====== Custom Classes ====== */
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6c63ff, #a78bfa, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0;
        line-height: 1.2;
    }

    .hero-subtitle {
        font-size: 1.1rem;
        color: var(--text-secondary);
        margin-top: 8px;
        font-weight: 400;
    }

    .feature-card {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 14px;
        padding: 24px;
        transition: all 0.3s ease;
        height: 100%;
    }

    .feature-card:hover {
        border-color: var(--accent-primary);
        box-shadow: var(--glow);
        transform: translateY(-2px);
    }

    .feature-icon {
        font-size: 2rem;
        margin-bottom: 12px;
    }

    .feature-title {
        font-size: 1rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 6px;
    }

    .feature-desc {
        font-size: 0.85rem;
        color: var(--text-secondary);
        line-height: 1.5;
    }

    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(63, 185, 80, 0.12);
        color: var(--success);
        border: 1px solid rgba(63, 185, 80, 0.25);
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 0.8rem;
        font-weight: 500;
    }

    .answer-container {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 14px;
        padding: 28px;
        margin-top: 16px;
        box-shadow: var(--glow);
    }

    .answer-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 16px;
        padding-bottom: 12px;
        border-bottom: 1px solid var(--border-color);
    }

    .answer-header-text {
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--accent-secondary);
    }

    .answer-body {
        color: var(--text-primary);
        line-height: 1.8;
        font-size: 0.95rem;
    }

    .source-chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(108, 99, 255, 0.1);
        border: 1px solid rgba(108, 99, 255, 0.25);
        border-radius: 8px;
        padding: 8px 14px;
        margin: 4px;
        font-size: 0.82rem;
        color: var(--accent-secondary);
        transition: all 0.2s ease;
        word-break: break-all;
    }

    .source-chip:hover {
        background: rgba(108, 99, 255, 0.2);
        border-color: var(--accent-primary);
    }

    .sidebar-section {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
    }

    .sidebar-label {
        font-size: 0.75rem;
        font-weight: 600;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-bottom: 10px;
    }

    .processing-step {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 10px 0;
        color: var(--text-secondary);
        font-size: 0.9rem;
    }

    .processing-step .step-icon {
        font-size: 1.2rem;
    }

    .stats-row {
        display: flex;
        gap: 12px;
        margin-top: 16px;
    }

    .stat-item {
        flex: 1;
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }

    .stat-value {
        font-size: 1.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6c63ff, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .stat-label {
        font-size: 0.75rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-top: 4px;
    }

    /* ====== Animated Gradient Border ====== */
    .gradient-border {
        position: relative;
        border-radius: 14px;
        padding: 1px;
        background: var(--accent-gradient);
    }

    .gradient-border-inner {
        background: var(--bg-card);
        border-radius: 13px;
        padding: 24px;
    }

    /* ====== Animations ====== */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }

    .animate-fade-in {
        animation: fadeInUp 0.6s ease-out;
    }

    .animate-pulse {
        animation: pulse 2s infinite;
    }
</style>
""", unsafe_allow_html=True)


# --- Sidebar ---
with st.sidebar:
    # Logo / Brand
    st.markdown("""
    <div style="text-align: center; padding: 16px 0 8px;">
        <div style="font-size: 2.5rem;">🧠</div>
        <div style="font-size: 1.3rem; font-weight: 700; 
             background: linear-gradient(135deg, #6c63ff, #a78bfa, #c084fc);
             -webkit-background-clip: text; -webkit-text-fill-color: transparent;
             background-clip: text; margin-top: 4px;">
            RockyBot
        </div>
        <div style="font-size: 0.75rem; color: #8b949e; margin-top: 2px;">
            AI-Powered News Research
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # URL Input Section
    st.markdown('<div class="sidebar-label">📎 News Article URLs</div>', unsafe_allow_html=True)

    urls = []
    for i in range(3):
        url = st.text_input(
            f"URL {i+1}",
            placeholder=f"https://example.com/article-{i+1}",
            label_visibility="collapsed",
            key=f"url_{i}"
        )
        urls.append(url)

    st.markdown("<div style='height: 8px'></div>", unsafe_allow_html=True)

    process_url_clicked = st.button("🚀  Process URLs", use_container_width=True)

    st.markdown("---")

    # Info section
    st.markdown("""
    <div class="sidebar-section">
        <div class="sidebar-label">💡 How it works</div>
        <div style="font-size: 0.82rem; color: #8b949e; line-height: 1.6;">
            <b>1.</b> Paste up to 3 news article URLs<br>
            <b>2.</b> Click <b>Process URLs</b> to analyze<br>
            <b>3.</b> Ask any question about the articles
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align: center; padding-top: 12px;">
        <div style="font-size: 0.7rem; color: #484f58;">
            Powered by <b style="color: #a78bfa;">Gemini 2.0 Flash</b> · FAISS
        </div>
    </div>
    """, unsafe_allow_html=True)


# --- LLM Initialization ---
# CHANGED: Initialize the Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.0,
    convert_system_message_to_human=True # Helps with compatibility for some chains
)

# --- Check for Session State Initialization ---
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "processed_urls" not in st.session_state:
    st.session_state.processed_urls = []
if "num_chunks" not in st.session_state:
    st.session_state.num_chunks = 0


# --- Main Content Area ---

# Hero Section
st.markdown("""
<div class="animate-fade-in">
    <div class="hero-title">News Research Assistant</div>
    <div class="hero-subtitle">
        Ask intelligent questions about any news article — powered by AI and semantic search.
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height: 24px'></div>", unsafe_allow_html=True)


# --- Feature cards (shown when no retriever is active) ---
if not st.session_state.retriever and not process_url_clicked:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="feature-card animate-fade-in" style="animation-delay: 0.1s">
            <div class="feature-icon">🔗</div>
            <div class="feature-title">Multi-URL Support</div>
            <div class="feature-desc">
                Analyze up to 3 news articles simultaneously for comprehensive research.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card animate-fade-in" style="animation-delay: 0.2s">
            <div class="feature-icon">🧠</div>
            <div class="feature-title">Semantic Search</div>
            <div class="feature-desc">
                FAISS vector store ensures accurate, context-aware answers from your articles.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="feature-card animate-fade-in" style="animation-delay: 0.3s">
            <div class="feature-icon">⚡</div>
            <div class="feature-title">Gemini Powered</div>
            <div class="feature-desc">
                Google's Gemini 2.0 Flash delivers fast, high-quality responses with source citations.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 32px'></div>", unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align: center; padding: 32px 0;">
        <div style="color: #484f58; font-size: 0.9rem;">
            👈 Paste your news article URLs in the sidebar to get started
        </div>
    </div>
    """, unsafe_allow_html=True)


# --- URL Processing ---
main_placeholder = st.empty()

if process_url_clicked:
    # Remove empty URLs
    urls = [url for url in urls if url.strip()]
    if not urls:
        st.sidebar.error("⚠️ Please provide at least one URL.")
    else:
        try:
            # Step 1: Load data
            main_placeholder.markdown("""
            <div class="processing-step animate-fade-in">
                <span class="step-icon">📥</span>
                <span>Loading article data...</span>
            </div>
            """, unsafe_allow_html=True)
            loader = UnstructuredURLLoader(urls=urls)
            data = loader.load()

            # Step 2: Split text
            main_placeholder.markdown("""
            <div class="processing-step animate-fade-in">
                <span class="step-icon">✂️</span>
                <span>Splitting text into chunks...</span>
            </div>
            """, unsafe_allow_html=True)
            text_splitter = RecursiveCharacterTextSplitter(
                separators=['\n\n', '\n', '.', ','],
                chunk_size=400,
                chunk_overlap=100
            )
            docs = text_splitter.split_documents(data)

            # Step 3: Create embeddings & FAISS vector store
            main_placeholder.markdown("""
            <div class="processing-step animate-fade-in">
                <span class="step-icon">🧬</span>
                <span>Building embedding vector store...</span>
            </div>
            """, unsafe_allow_html=True)
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
            vectorstore = FAISS.from_documents(docs, embeddings)
            time.sleep(1)

            # Save retriever to session state
            st.session_state.retriever = vectorstore.as_retriever()
            st.session_state.processed_urls = urls
            st.session_state.num_chunks = len(docs)
            main_placeholder.empty()

            st.markdown("""
            <div class="gradient-border animate-fade-in">
                <div class="gradient-border-inner" style="text-align: center;">
                    <div style="font-size: 2rem; margin-bottom: 8px;">✅</div>
                    <div style="font-size: 1.1rem; font-weight: 600; color: #3fb950;">
                        URLs Processed Successfully!
                    </div>
                    <div style="font-size: 0.85rem; color: #8b949e; margin-top: 6px;">
                        Your articles are ready. Ask any question below.
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            main_placeholder.empty()
            st.markdown(f"""
            <div class="answer-container" style="border-color: rgba(248, 81, 73, 0.3);">
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px;">
                    <span style="font-size: 1.3rem;">❌</span>
                    <span style="font-weight: 600; color: #f85149;">Error Processing URLs</span>
                </div>
                <div style="color: #8b949e; font-size: 0.9rem; line-height: 1.6;">
                    {e}<br><br>
                    Please check that your URLs are valid and accessible.
                </div>
            </div>
            """, unsafe_allow_html=True)


# --- The Modern RAG Chain (LCEL) ---
prompt = ChatPromptTemplate.from_template("""
Answer the user's question based ONLY on the following context.
If the answer is not in the context, say you don't know.
Provide a detailed answer and list the sources used.

Context:
{context}

Question:
{input}
""")

document_chain = create_stuff_documents_chain(llm, prompt)

if st.session_state.retriever:
    retrieval_chain = create_retrieval_chain(st.session_state.retriever, document_chain)

    # Stats row
    st.markdown(f"""
    <div class="stats-row animate-fade-in">
        <div class="stat-item">
            <div class="stat-value">{len(st.session_state.processed_urls)}</div>
            <div class="stat-label">Articles</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">{st.session_state.num_chunks}</div>
            <div class="stat-label">Chunks</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">
                <span style="font-size: 1.5rem;">⚡</span>
            </div>
            <div class="stat-label">Ready</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height: 24px'></div>", unsafe_allow_html=True)

    query = st.text_input(
        "Ask a Question",
        placeholder="What are the key takeaways from these articles?",
        label_visibility="collapsed"
    )

    if query:
        with st.spinner(""):
            # Show a custom loading state
            loading_placeholder = st.empty()
            loading_placeholder.markdown("""
            <div class="processing-step animate-pulse">
                <span class="step-icon">🔍</span>
                <span style="color: #a78bfa; font-weight: 500;">Searching and analyzing articles...</span>
            </div>
            """, unsafe_allow_html=True)

            result = retrieval_chain.invoke({"input": query})
            loading_placeholder.empty()

        # Display answer in styled container
        st.markdown(f"""
        <div class="answer-container animate-fade-in">
            <div class="answer-header">
                <span style="font-size: 1.3rem;">💬</span>
                <span class="answer-header-text">Answer</span>
            </div>
            <div class="answer-body">
                {result["answer"]}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Display sources
        sources = result.get("context", [])
        if sources:
            unique_sources = set(doc.metadata.get('source', 'Unknown') for doc in sources)
            st.markdown("<div style='height: 16px'></div>", unsafe_allow_html=True)
            st.markdown("""
            <div style="font-size: 0.85rem; font-weight: 600; color: #8b949e; 
                 text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px;">
                📄 Sources
            </div>
            """, unsafe_allow_html=True)

            sources_html = ""
            for source in unique_sources:
                sources_html += f'<span class="source-chip">🔗 {source}</span>'

            st.markdown(f"""
            <div class="animate-fade-in" style="display: flex; flex-wrap: wrap; gap: 4px;">
                {sources_html}
            </div>
            """, unsafe_allow_html=True)