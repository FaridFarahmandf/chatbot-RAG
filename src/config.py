"""
Central configuration for the RAG chatbot.

All settings are read from environment variables (loaded from a `.env` file
locally), then from Streamlit secrets (on Streamlit Community Cloud), with
sensible defaults so the project runs out of the box.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load variables from a .env file in the project root, if present.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _get(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None:
        # On Streamlit Community Cloud, settings are entered as app secrets.
        try:
            import streamlit as st

            value = st.secrets.get(name)
        except Exception:  # noqa: BLE001 - no secrets file, or not in Streamlit
            value = None
    return default if value is None else str(value)


@dataclass
class Config:
    """Runtime configuration, populated from environment variables."""

    # LLM backend: "ollama" (local) or "groq" (free cloud)
    llm_backend: str = _get("LLM_BACKEND", "ollama").lower()

    # Ollama
    ollama_model: str = _get("OLLAMA_MODEL", "llama3.1:8b")
    ollama_base_url: str = _get("OLLAMA_BASE_URL", "http://localhost:11434")

    # Groq
    groq_api_key: str = _get("GROQ_API_KEY", "")
    groq_model: str = _get("GROQ_MODEL", "qwen/qwen3.8-27b")

    # Embeddings (local, free)
    embedding_model: str = _get(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )

    # Retrieval / chunking
    chunk_size: int = int(_get("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(_get("CHUNK_OVERLAP", "150"))
    top_k: int = int(_get("TOP_K", "4"))

    # Show "Add documents" and "Rebuild index" in the web UI. Off by default so
    # visitors to a deployed app can't change the knowledge base; set
    # ADMIN_TOOLS=true in your local .env to get them back.
    admin_tools: bool = _get("ADMIN_TOOLS", "false").lower() == "true"

    # Paths (resolved to absolute, relative to project root)
    data_dir: Path = PROJECT_ROOT / _get("DATA_DIR", "data")
    vector_store_dir: Path = PROJECT_ROOT / _get("VECTOR_STORE_DIR", "vector_store")


# A single shared instance imported across the project.
config = Config()
