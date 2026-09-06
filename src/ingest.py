"""
Document ingestion pipeline.

Steps:
  1. Load documents from the data directory (PDF, TXT, Markdown).
  2. Split them into overlapping chunks.
  3. Embed each chunk with a local (free) sentence-transformers model.
  4. Store the vectors in a persistent Chroma vector store.

Run directly to (re)build the index:
    python -m src.ingest
"""

from __future__ import annotations

import shutil
from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import config
from .vector_store import get_embeddings, build_vector_store

# File extensions we know how to load.
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def load_documents(data_dir: Path) -> list[Document]:
    """Load every supported file in `data_dir` into LangChain Documents."""
    documents: list[Document] = []

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}. "
            "Create it and add some .pdf/.txt/.md files."
        )

    files = sorted(
        p for p in data_dir.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not files:
        raise ValueError(
            f"No supported documents found in {data_dir}. "
            f"Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    for path in files:
        suffix = path.suffix.lower()
        try:
            if suffix == ".pdf":
                loader = PyPDFLoader(str(path))
            else:  # .txt or .md
                loader = TextLoader(str(path), encoding="utf-8")

            loaded = loader.load()
            # Tag each doc with a clean source name for citation in the UI.
            for doc in loaded:
                doc.metadata["source"] = path.name
            documents.extend(loaded)
            print(f"  Loaded {path.name} ({len(loaded)} page(s)/section(s))")
        except Exception as exc:  # noqa: BLE001 - report and keep going
            print(f"  ! Skipped {path.name}: {exc}")

    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """Split documents into overlapping chunks for retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        # Split on paragraph/line boundaries first, then fall back to words.
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    return chunks


def ingest(rebuild: bool = True) -> int:
    """
    Full ingestion pipeline. Returns the number of chunks indexed.

    If `rebuild` is True, any existing vector store is deleted first so the
    index always reflects the current contents of the data directory.
    """
    print(f"Ingesting documents from: {config.data_dir}")

    if rebuild and config.vector_store_dir.exists():
        print("Removing previous vector store...")
        shutil.rmtree(config.vector_store_dir)

    documents = load_documents(config.data_dir)
    print(f"Loaded {len(documents)} document section(s). Splitting into chunks...")

    chunks = split_documents(documents)
    print(f"Created {len(chunks)} chunk(s). Embedding + indexing (first run "
          "downloads the embedding model, please wait)...")

    embeddings = get_embeddings()
    build_vector_store(chunks, embeddings)

    print(f"Done. Indexed {len(chunks)} chunk(s) into {config.vector_store_dir}")
    return len(chunks)


if __name__ == "__main__":
    ingest(rebuild=True)
