"""
Vector store helpers.

Uses a local sentence-transformers embedding model (free, no API key) and a
persistent Chroma database on disk. Kept separate so both the ingestion script
and the chat engine share exactly the same embedding + store configuration.
"""

from __future__ import annotations

import shutil

from chromadb.api.shared_system_client import SharedSystemClient
from chromadb.config import Settings
from chromadb.telemetry.product import ProductTelemetryClient, ProductTelemetryEvent
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from overrides import override

from .config import config

# Collection name inside Chroma. Keeping it fixed lets ingest + query agree.
COLLECTION_NAME = "rag_documents"


class NoOpTelemetry(ProductTelemetryClient):
    """Telemetry client that sends nothing.

    Chroma's built-in client still calls posthog when telemetry is disabled,
    which logs "Failed to send telemetry event" on newer posthog versions.
    """

    @override
    def capture(self, event: ProductTelemetryEvent) -> None:
        pass


def _client_settings() -> Settings:
    """Chroma client settings with anonymous usage telemetry turned off."""
    return Settings(
        anonymized_telemetry=False,
        chroma_product_telemetry_impl=f"{__name__}.{NoOpTelemetry.__name__}",
        # Must be set explicitly: langchain-chroma only sets it when no
        # client_settings are passed, and without it the store is in-memory.
        is_persistent=True,
    )


def get_embeddings() -> HuggingFaceEmbeddings:
    """Return the local embedding model. Downloaded once, then cached."""
    return HuggingFaceEmbeddings(
        model_name=config.embedding_model,
        # Normalizing helps cosine-similarity retrieval quality.
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vector_store(
    chunks: list[Document], embeddings: HuggingFaceEmbeddings
) -> Chroma:
    """Create (and persist) a Chroma store from document chunks."""
    config.vector_store_dir.mkdir(parents=True, exist_ok=True)
    store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(config.vector_store_dir),
        client_settings=_client_settings(),
    )
    return store


def delete_vector_store() -> None:
    """Delete the persisted store from disk, if there is one."""
    # Chroma caches one client per directory for the life of the process.
    # Without dropping it, a rebuild in a long-running process (the Streamlit
    # app) reuses a client whose files are gone and fails to connect.
    SharedSystemClient.clear_system_cache()
    if config.vector_store_dir.exists():
        shutil.rmtree(config.vector_store_dir)


def load_vector_store(embeddings: HuggingFaceEmbeddings | None = None) -> Chroma:
    """Open the existing persisted Chroma store for querying."""
    if embeddings is None:
        embeddings = get_embeddings()

    if not config.vector_store_dir.exists():
        raise FileNotFoundError(
            "No vector store found. Run `python -m src.ingest` first "
            "(or click 'Rebuild index' in the app)."
        )

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(config.vector_store_dir),
        client_settings=_client_settings(),
    )
