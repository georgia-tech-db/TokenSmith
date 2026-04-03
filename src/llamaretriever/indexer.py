"""
Document ingestion, parsing, contextual chunk enrichment, and index management.

Pipeline:
  1. Load markdown docs from data/
  2. Parse with MarkdownNodeParser
  3. Apply SentenceSplitter
  4. Extract chapter / section / subsection hierarchy from markdown headers
  5. Prepend contextual prefix to each chunk before indexing
  6. Build VectorStoreIndex with GGUF embeddings
  7. Persist to disk
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Iterable

from llama_index.core import (
    Document,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.core.schema import BaseNode

from .config import LlamaIndexConfig


_HEADER_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


def load_markdown_documents(data_dir: str) -> list[Document]:
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
    return IngestionPipeline(
        transformations=[
            MarkdownNodeParser(),
            SentenceSplitter(
                chunk_size=cfg.chunk_size,
                chunk_overlap=cfg.chunk_overlap,
            ),
        ]
    )


def _clean_header_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    return text.strip(" -\u2013\u2014:")


def _extract_header_path(text: str) -> list[str]:
    """Extract the deepest heading path from the chunk's markdown headers."""
    headers = []
    for hashes, title in _HEADER_RE.findall(text):
        title = _clean_header_text(title)
        if title:
            headers.append((len(hashes), title))

    if not headers:
        return []

    path: list[tuple[int, str]] = []
    for level, title in headers:
        while path and path[-1][0] >= level:
            path.pop()
        path.append((level, title))

    return [title for _, title in path]


def _path_to_hierarchy(path: list[str]) -> tuple[str, str, str]:
    chapter = path[0] if len(path) >= 1 else "Unknown"
    section = path[1] if len(path) >= 2 else chapter
    subsection = path[2] if len(path) >= 3 else section
    return chapter, section, subsection


def _build_context_prefix(
    *,
    source: str,
    chapter: str,
    section: str,
    subsection: str,
    header_path: list[str],
    max_chars: int,
) -> str:
    header_path_str = " > ".join(header_path) if header_path else chapter
    prefix = (
        f"[Source: {source}]\n"
        f"[Chapter: {chapter}]\n"
        f"[Section: {section}]\n"
        f"[Subsection: {subsection}]\n"
        f"[Header Path: {header_path_str}]\n"
        f"[Context: This chunk comes from a database textbook passage under the above "
        f"document hierarchy. Use that hierarchy to interpret the passage precisely.]\n\n"
    )
    if len(prefix) > max_chars:
        prefix = prefix[: max_chars - 3].rstrip() + "...\n\n"
    return prefix


def enrich_nodes_with_context(nodes: Iterable[BaseNode], cfg: LlamaIndexConfig) -> list[BaseNode]:
    enriched_nodes: list[BaseNode] = []

    for node_idx, node in enumerate(nodes, start=1):
        raw_text = node.text or ""
        source = (
            node.metadata.get("file_name")
            or node.metadata.get("filename")
            or node.metadata.get("source")
            or "chunk"
        )

        header_path = _extract_header_path(raw_text)
        chapter, section, subsection = _path_to_hierarchy(header_path if header_path else ["Unknown"])

        node.metadata["chapter"] = chapter
        node.metadata["section"] = section
        node.metadata["subsection"] = subsection
        node.metadata["header_path"] = " > ".join(header_path) if header_path else chapter
        node.metadata["source"] = source
        node.metadata["raw_text"] = raw_text
        node.metadata["node_seq"] = node_idx

        if cfg.enrich_chunks_with_context:
            prefix = _build_context_prefix(
                source=source,
                chapter=chapter,
                section=section,
                subsection=subsection,
                header_path=header_path,
                max_chars=cfg.max_context_prefix_chars,
            )
            node.text = prefix + raw_text
            node.metadata["context_prefix"] = prefix
        else:
            node.metadata["context_prefix"] = ""

        enriched_nodes.append(node)

    return enriched_nodes


def build_index(cfg: LlamaIndexConfig) -> VectorStoreIndex:
    print("=" * 60)
    print("Building LlamaIndex VectorStoreIndex ...")
    print(f"  Data dir    : {cfg.data_dir}")
    print(f"  Persist dir : {cfg.persist_dir}")
    print(f"  Embed model : {cfg.embed_model}")
    print(f"  Chunk size  : {cfg.chunk_size}  overlap: {cfg.chunk_overlap}")
    print("=" * 60)

    t0 = time.time()

    documents = load_markdown_documents(cfg.data_dir)
    print(f"Loaded {len(documents)} document(s) in {time.time() - t0:.1f}s")

    pipeline = build_ingestion_pipeline(cfg)
    nodes = pipeline.run(documents=documents, show_progress=True)
    print(f"Created {len(nodes)} nodes after parsing + chunking")

    nodes = enrich_nodes_with_context(nodes, cfg)
    print(f"Enriched {len(nodes)} nodes with chapter/section/subsection context")

    t1 = time.time()
    index = VectorStoreIndex(nodes, show_progress=True)
    print(f"Index built in {time.time() - t1:.1f}s")

    index.storage_context.persist(persist_dir=cfg.persist_dir)
    print(f"Index persisted to {cfg.persist_dir}")
    print(f"Total indexing time: {time.time() - t0:.1f}s")

    return index


def load_index(cfg: LlamaIndexConfig) -> VectorStoreIndex:
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
    persist_path = Path(cfg.persist_dir)
    if not force_rebuild and persist_path.exists() and any(persist_path.iterdir()):
        return load_index(cfg)
    return build_index(cfg)
