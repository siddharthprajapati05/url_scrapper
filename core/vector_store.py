"""
Vector Store Manager module.
Handles FAISS vector store creation, persistence, and retrieval with caching.
"""

import os
import logging
import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from config import AppConfig

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Manages FAISS vector store lifecycle."""

    STORE_DIR = "faiss_store"

    def __init__(self, config: AppConfig):
        self.config = config

    @staticmethod
    @st.cache_resource(show_spinner=False)
    def get_embeddings(model_name: str):
        """
        Load and cache the embedding model.
        Only loaded once regardless of how many times Streamlit reruns.
        """
        logger.info(f"Loading embedding model: {model_name}")
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    def create_store(self, chunks) -> FAISS:
        """Create a FAISS vector store from document chunks."""
        embeddings = self.get_embeddings(self.config.embedding_model)
        vectorstore = FAISS.from_documents(chunks, embeddings)
        logger.info(f"Created FAISS store with {len(chunks)} vectors")
        return vectorstore

    def get_retriever(self, vectorstore: FAISS):
        """Create a retriever from the vector store."""
        return vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.config.retriever_top_k},
        )

    def save_store(self, vectorstore: FAISS, path: str | None = None):
        """Persist the vector store to disk."""
        save_path = path or self.STORE_DIR
        vectorstore.save_local(save_path)
        logger.info(f"Saved vector store to {save_path}")

    def load_store(self, path: str | None = None) -> FAISS | None:
        """Load a persisted vector store from disk."""
        load_path = path or self.STORE_DIR
        if not os.path.exists(load_path):
            return None
        try:
            embeddings = self.get_embeddings(self.config.embedding_model)
            vectorstore = FAISS.load_local(
                load_path, embeddings, allow_dangerous_deserialization=True
            )
            logger.info(f"Loaded vector store from {load_path}")
            return vectorstore
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            return None
