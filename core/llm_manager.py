"""
LLM Manager module.
Handles LLM initialization with caching, model switching, and retry logic.
"""

import logging
import streamlit as st

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from config import AppConfig, ModelConfig

logger = logging.getLogger(__name__)


# --- Prompt Templates ---

RESEARCH_PROMPT = ChatPromptTemplate.from_template("""
You are an expert news research analyst. Answer the user's question based ONLY on the provided context.

Rules:
1. Use ONLY the information from the context below. Do not add external knowledge.
2. If the answer is not in the context, clearly state: "I couldn't find this information in the provided articles."
3. Be detailed, accurate, and well-structured in your response.
4. Use bullet points or numbered lists for clarity when appropriate.
5. Always mention which source(s) the information comes from.

Context:
{context}

Question:
{input}

Provide a comprehensive, well-structured answer:
""")


class LLMManager:
    """Manages LLM initialization, chain creation, and inference."""

    def __init__(self, config: AppConfig):
        self.config = config

    @staticmethod
    @st.cache_resource(show_spinner=False)
    def get_llm(model_id: str, api_key: str, temperature: float):
        """
        Initialize and cache the LLM.
        Uses st.cache_resource so the model is only loaded once per config.
        """
        logger.info(f"Initializing LLM: {model_id}")
        return ChatGoogleGenerativeAI(
            model=model_id,
            google_api_key=api_key,
            temperature=temperature,
            max_output_tokens=2048,
            convert_system_message_to_human=True,
        )

    def create_llm(self, model_config: ModelConfig):
        """Create an LLM instance for the given model config."""
        return self.get_llm(
            model_id=model_config.model_id,
            api_key=self.config.google_api_key,
            temperature=self.config.temperature,
        )

    @staticmethod
    def build_chain(llm, retriever):
        """Build the RAG retrieval chain."""
        document_chain = create_stuff_documents_chain(llm, RESEARCH_PROMPT)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        return retrieval_chain

    @staticmethod
    def format_sources(result: dict) -> list[dict]:
        """Extract and deduplicate sources from the result."""
        context_docs = result.get("context", [])
        seen = set()
        sources = []
        for doc in context_docs:
            # Try multiple metadata keys — different loaders use different keys
            source = (
                doc.metadata.get("source")
                or doc.metadata.get("url")
                or doc.metadata.get("file_path")
            )
            if source and source not in seen and source != "Unknown":
                seen.add(source)
                sources.append({
                    "url": source,
                    "snippet": doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content,
                })
        return sources
