"""
Application configuration module.
Centralizes all settings, constants, and environment variable management.
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for a single LLM model."""
    name: str
    model_id: str
    provider: str
    description: str


# Available models — add new ones here
AVAILABLE_MODELS: list[ModelConfig] = [
    ModelConfig("Gemini 2.5 Flash", "gemini-2.5-flash", "google", "Latest & fastest"),
    ModelConfig("Gemini 2.0 Flash", "gemini-2.0-flash", "google", "Stable & reliable"),
    ModelConfig("Gemini 2.5 Pro", "gemini-2.5-pro-preview-06-05", "google", "Most capable"),
]


@dataclass
class AppConfig:
    """Central application configuration."""

    # --- API Keys ---
    google_api_key: str = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""))

    # --- Text Splitting ---
    chunk_size: int = 500
    chunk_overlap: int = 100
    separators: list[str] = field(default_factory=lambda: ["\n\n", "\n", ". ", ", ", " "])

    # --- Embeddings ---
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"

    # --- Retriever ---
    retriever_top_k: int = 5

    # --- LLM ---
    temperature: float = 0.1
    max_output_tokens: int = 2048

    # --- App ---
    max_urls: int = 3
    app_title: str = "LexaRAG"
    app_description: str = "AI-Powered News Research Assistant"

    def validate(self) -> list[str]:
        """Validate configuration. Returns list of error messages."""
        errors = []
        if not self.google_api_key:
            errors.append("GOOGLE_API_KEY is not set in .env file.")
        if self.chunk_size < 100:
            errors.append("chunk_size should be at least 100.")
        if self.retriever_top_k < 1:
            errors.append("retriever_top_k must be >= 1.")
        return errors


# Singleton config instance
config = AppConfig()
