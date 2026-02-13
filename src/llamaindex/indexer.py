"""
Document ingestion, parsing, and index management.

Pipeline:
  1. Load markdown docs from data/
  2. Parse with MarkdownNodeParser (header-aware splitting)
  3. Apply SentenceSplitter for size-consistent chunks
  4. Build VectorStoreIndex with HuggingFace embeddings
  5. Persist to disk for fast reload
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from llama_index.core import (
    Document,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
    Settings,
)
from llama_index.core.node_parser import (
    MarkdownNodeParser,
    SentenceSplitter,
)
from llama_index.core.ingestion import IngestionPipeline

from .config import LlamaIndexConfig


def load_markdown_documents(data_dir: str) -> list[Document]:
    """Load all markdown files from data_dir."""
    data_path = Path(data_dir)
    md_files = sorted(data_path.glob("*.md"))
    if not md_files:
        raise FileNotFoundError(
            f"No markdown files found in {data_dir}/. "
            "Run extraction first or place .md files there."
        )
    print(f"Found {len(md_files)} markdown file(s): {[f.name for f in md_files]}")
    reader = SimpleDirectoryReader(input_files=[str(f) for f in md_files])
    return reader.load_data(show_progress=True)


def build_ingestion_pipeline(cfg: LlamaIndexConfig) -> IngestionPipeline:
    """
    Build a two-stage parsing pipeline:
      1. MarkdownNodeParser — splits on markdown headers, preserving structure
      2. SentenceSplitter  — enforces max chunk size with overlap
    """
    return IngestionPipeline(
        transformations=[
            MarkdownNodeParser(),
            SentenceSplitter(
                chunk_size=cfg.chunk_size,
                chunk_overlap=cfg.chunk_overlap,
            ),
        ]
    )


def build_index(cfg: LlamaIndexConfig) -> VectorStoreIndex:
    """
    Build a fresh VectorStoreIndex from documents and persist it.

    Returns the ready-to-query index.
    """
    print("=" * 60)
    print("Building LlamaIndex VectorStoreIndex ...")
    print(f"  Data dir    : {cfg.data_dir}")
    print(f"  Persist dir : {cfg.persist_dir}")
    print(f"  Embed model : {cfg.embed_model_name}")
    print(f"  Chunk size  : {cfg.chunk_size}  overlap: {cfg.chunk_overlap}")
    print("=" * 60)

    t0 = time.time()

    # Step 1: Load documents
    documents = load_markdown_documents(cfg.data_dir)
    print(f"Loaded {len(documents)} document(s) in {time.time() - t0:.1f}s")

    # Step 2: Ingest (parse + chunk)
    pipeline = build_ingestion_pipeline(cfg)
    nodes = pipeline.run(documents=documents, show_progress=True)
    print(f"Created {len(nodes)} nodes after parsing + chunking")

    # Step 3: Build vector index
    t1 = time.time()
    index = VectorStoreIndex(
        nodes,
        show_progress=True,
    )
    print(f"Index built in {time.time() - t1:.1f}s")

    # Step 4: Persist
    index.storage_context.persist(persist_dir=cfg.persist_dir)
    print(f"Index persisted to {cfg.persist_dir}")
    print(f"Total indexing time: {time.time() - t0:.1f}s")

    return index


def load_index(cfg: LlamaIndexConfig) -> VectorStoreIndex:
    """Load a previously persisted index from disk."""
    persist_path = Path(cfg.persist_dir)
    if not persist_path.exists():
        raise FileNotFoundError(
            f"No persisted index at {cfg.persist_dir}. Run indexing first."
        )
    print(f"Loading index from {cfg.persist_dir} ...")
    storage_context = StorageContext.from_defaults(persist_dir=cfg.persist_dir)
    index = load_index_from_storage(storage_context)
    print("Index loaded successfully.")
    return index


def get_or_build_index(cfg: LlamaIndexConfig, force_rebuild: bool = False) -> VectorStoreIndex:
    """Load existing index or build a new one."""
    persist_path = Path(cfg.persist_dir)
    if not force_rebuild and persist_path.exists() and any(persist_path.iterdir()):
        return load_index(cfg)
    return build_index(cfg)
