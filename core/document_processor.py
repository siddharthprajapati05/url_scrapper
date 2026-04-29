"""
Document processing module.
Handles URL validation, loading, and text splitting with robust error handling.
"""

import re
import logging
from urllib.parse import urlparse

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader

from config import AppConfig

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Handles loading and processing of news article documents."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            separators=config.separators,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

    @staticmethod
    def validate_url(url: str) -> bool:
        """Validate that a string is a proper HTTP/HTTPS URL."""
        try:
            result = urlparse(url.strip())
            return all([result.scheme in ("http", "https"), result.netloc])
        except Exception:
            return False

    @staticmethod
    def clean_urls(urls: list[str]) -> list[str]:
        """Filter and clean a list of URLs."""
        cleaned = []
        for url in urls:
            url = url.strip()
            if url and DocumentProcessor.validate_url(url):
                cleaned.append(url)
        return list(dict.fromkeys(cleaned))  # Remove duplicates, preserve order

    def load_documents(self, urls: list[str]):
        """
        Load documents from URLs with error handling.
        Returns (documents, errors) tuple.
        """
        valid_urls = self.clean_urls(urls)
        if not valid_urls:
            return [], ["No valid URLs provided."]

        errors = []
        try:
            loader = UnstructuredURLLoader(
                urls=valid_urls,
                show_progress_bar=False,
            )
            documents = loader.load()

            if not documents:
                errors.append("No content could be extracted from the provided URLs.")
                return [], errors

            logger.info(f"Loaded {len(documents)} document elements from {len(valid_urls)} URLs")
            return documents, errors

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error loading documents: {error_msg}")

            # Provide user-friendly error messages
            if "SSL" in error_msg or "certificate" in error_msg.lower():
                errors.append("SSL certificate error. The website may have security issues.")
            elif "ConnectionError" in error_msg or "timeout" in error_msg.lower():
                errors.append("Could not connect to one or more URLs. Please check your internet connection.")
            elif "404" in error_msg:
                errors.append("One or more URLs returned a 404 Not Found error.")
            else:
                errors.append(f"Failed to load articles: {error_msg}")

            return [], errors

    def split_documents(self, documents):
        """Split documents into chunks for embedding."""
        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks")
        return chunks

    def process(self, urls: list[str]):
        """
        Full pipeline: validate → load → split.
        Returns (chunks, metadata, errors).
        """
        valid_urls = self.clean_urls(urls)
        invalid_urls = [u for u in urls if u.strip() and u.strip() not in valid_urls]

        documents, errors = self.load_documents(valid_urls)

        if not documents:
            return [], {}, errors

        chunks = self.split_documents(documents)

        metadata = {
            "total_urls": len(valid_urls),
            "invalid_urls": invalid_urls,
            "total_documents": len(documents),
            "total_chunks": len(chunks),
            "avg_chunk_size": sum(len(c.page_content) for c in chunks) // max(len(chunks), 1),
            "sources": list(set(doc.metadata.get("source", "Unknown") for doc in documents)),
        }

        return chunks, metadata, errors
