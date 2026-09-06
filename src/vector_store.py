"""
Vector store helpers.

Uses a local sentence-transformers embedding model (free, no API key) and a
persistent Chroma database on disk. Kept separate so both the ingestion script
and the chat engine share exactly the same embedding + store configuration.
"""

from __future__ import annotations

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from .config import config

# Collection name inside Chroma. Keeping it fixed lets ingest + query agree.
COLLECTION_NAME = "rag_documents"


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
    )
    return store


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
    )
