<div align="center">

# 🧠 LexaRAG

**AI-Powered News Research Assistant**

Paste news article URLs or upload PDFs — ask questions, get cited answers powered by RAG.

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Google Gemini](https://img.shields.io/badge/Gemini-2.5_Flash-4285F4?style=for-the-badge&logo=googlegemini&logoColor=white)](https://ai.google.dev)
[![LangChain](https://img.shields.io/badge/LangChain-0.3.7-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://langchain.com)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-0467DF?style=for-the-badge&logo=meta&logoColor=white)](https://github.com/facebookresearch/faiss)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Embeddings-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co)
[![PyPDF](https://img.shields.io/badge/PyPDF-PDF_Support-EC1C24?style=for-the-badge&logo=adobeacrobatreader&logoColor=white)](https://pypdf.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

</div>

---

## ✨ Features

- 🔗 **Multi-URL Ingestion** — Analyze up to 3 news articles at once
- 📄 **PDF Upload** — Upload up to 3 PDFs alongside URLs
- 🧠 **Semantic Search** — FAISS + HuggingFace embeddings for precise retrieval
- ⚡ **Gemini AI** — Fast, accurate answers with source citations
- 💬 **Chat History** — Conversational memory for follow-up questions
- 🎨 **Premium Dark UI** — Custom styled with animations and gradients

---

## 📁 Project Structure

```
news/
├── main.py                   # Streamlit app — UI, chat, orchestration
├── config.py                 # Configuration & model registry
├── core/
│   ├── __init__.py
│   ├── document_processor.py # URL loading, PDF loading, text splitting
│   ├── llm_manager.py       # LLM init, RAG chain builder
│   └── vector_store.py      # FAISS vector store lifecycle
├── requirements.txt
├── .env                      # API keys (not committed)
└── README.md
```

---

## 🚀 Quick Start

**Prerequisites:** Python 3.10+ · [Google Gemini API Key](https://aistudio.google.com/apikey)

```bash
# Clone
git clone https://github.com/your-username/lexarag.git
cd lexarag

# Setup environment
python3 -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# Configure API key
echo 'GOOGLE_API_KEY="your_key_here"' > .env

# Run
streamlit run main.py
```

App opens at **http://localhost:8501**

---

## 🎯 How It Works

```mermaid
graph LR
    A[📎 URLs / PDFs] --> B[📥 Load Content]
    B --> C[✂️ Chunk Text]
    C --> D[🧬 Embed]
    D --> E[📦 FAISS Index]
    E --> F[🔍 Retrieve]
    F --> G[🤖 Gemini]
    G --> H[💬 Cited Answer]
```

1. Paste URLs and/or upload PDFs in the sidebar
2. Click **Process** — articles are loaded, chunked, and embedded
3. Ask questions in the chat
4. Get cited answers with source links (🔗 for URLs, 📄 for PDFs)

---

## ⚙️ Configuration

All settings in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `chunk_size` | `500` | Characters per chunk |
| `chunk_overlap` | `100` | Overlap between chunks |
| `embedding_model` | `all-mpnet-base-v2` | HuggingFace sentence transformer |
| `retriever_top_k` | `5` | Chunks retrieved per query |
| `temperature` | `0.1` | LLM randomness |
| `max_urls` | `3` | Max URL inputs |

---

## 🔧 Troubleshooting

| Issue | Fix |
|-------|-----|
| 429 Rate Limit | Wait 60s or use a new API key |
| 404 Model Not Found | Update `model_id` in `config.py` |
| Slow First Load | Embedding model (~420MB) downloads once |
| GOOGLE_API_KEY not set | Create `.env` — see Quick Start |

---

## 📝 License

MIT — see [LICENSE](LICENSE)

---

<div align="center">
  <b>Built with ❤️ by Siddharth Prajapati</b>
</div>
