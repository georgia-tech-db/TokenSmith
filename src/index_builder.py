#!/usr/bin/env python3
"""
index_builder.py
PDF -> markdown text -> contextualized chunks/sections -> embeddings -> BM25 + FAISS + metadata
"""

from __future__ import annotations

import json
import os
import pathlib
import pickle
import re
from typing import Any, Dict, List, Sequence

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from src.artifacts import artifact_file_map, build_manifest, save_manifest
from src.embedder import SentenceTransformer
from src.preprocessing.chunking import ChunkConfig, DocumentChunker
from src.preprocessing.extraction import load_extracted_sections

# ----- runtime parallelism knobs (avoid oversubscription) -----
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

DEFAULT_EXCLUSION_KEYWORDS = ["questions", "exercises", "summary", "references"]
PAGE_MARKER_RE = re.compile(r"--- Page (\d+) ---")


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


def _clean_page_markers(text: str) -> str:
    """Remove ``--- Page N ---`` markers from text and strip whitespace."""
    return re.sub(PAGE_MARKER_RE, "", text).strip()


def _contextualize_text(section_path: str, page_numbers: Sequence[int], content: str) -> str:
    """Wrap chunk content with its section path and page span for contextualized embedding."""
    pages = sorted({int(page) for page in page_numbers if isinstance(page, int)})
    if not pages:
        span = "unknown pages"
    elif len(pages) == 1:
        span = f"page {pages[0]}"
    else:
        span = f"pages {pages[0]}-{pages[-1]}"

    return (
        f"Section Path: {section_path}\n"
        f"Page Span: {span}\n"
        f"Content:\n{content.strip()}"
    )


def _extract_chunk_pages(sub_chunk: str, current_page: int) -> tuple[List[int], int]:
    """Parse page markers from a chunk and return (sorted_pages, updated_current_page)."""
    chunk_pages = set()
    fragments = PAGE_MARKER_RE.split(sub_chunk)

    if fragments[0].strip():
        chunk_pages.add(current_page)

    for index in range(1, len(fragments), 2):
        try:
            new_page = int(fragments[index]) + 1
        except ValueError:
            continue

        if index + 1 < len(fragments) and fragments[index + 1].strip():
            chunk_pages.add(new_page)
        current_page = new_page

    return sorted(chunk_pages), current_page


def _embed_texts(
    *,
    embedder: SentenceTransformer,
    texts: Sequence[str],
    use_multiprocessing: bool,
    batch_size: int,
) -> np.ndarray:
    """Encode texts into embeddings, optionally using a multiprocess pool."""
    if not texts:
        return np.empty((0, embedder.embedding_dimension), dtype=np.float32)

    if use_multiprocessing:
        print("Starting multi-process pool for embeddings...")
        pool = embedder.start_multi_process_pool(
            num_workers=_env_int("TOKENSMITH_EMBED_WORKERS", 2)
        )
        try:
            return embedder.encode_multi_process(list(texts), pool, batch_size=batch_size)
        finally:
            embedder.stop_multi_process_pool(pool)

    return embedder.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )


def _build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """Create a flat L2 FAISS index from a numpy embedding matrix."""
    if embeddings.size == 0:
        raise ValueError("Cannot build FAISS index with zero embeddings")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index


def _build_bm25_index(texts: Sequence[str]) -> BM25Okapi:
    """Build a BM25Okapi index from pre-tokenized texts."""
    tokenized_chunks = [preprocess_for_bm25(text) for text in texts]
    return BM25Okapi(tokenized_chunks)


def _extract_and_chunk(
    markdown_file: str,
    chunker: DocumentChunker,
    chunk_config: ChunkConfig,
) -> tuple[
    List[str], List[str], List[Dict[str, Any]],
    List[str], List[str], List[Dict[str, Any]],
    Dict[int, set[int]],
]:
    """Extract sections from a source document and chunk them into contextualized texts.

    Returns a tuple of (chunk_texts, chunk_sources, chunk_meta,
    section_texts, section_sources, section_meta, page_to_chunk_ids).
    """
    source_path = pathlib.Path(markdown_file)
    sections = load_extracted_sections(
        markdown_file,
        exclusion_keywords=DEFAULT_EXCLUSION_KEYWORDS,
    )
    if not sections:
        raise ValueError(f"No sections were extracted from {source_path}")

    chunk_texts: List[str] = []
    chunk_sources: List[str] = []
    chunk_meta: List[Dict[str, Any]] = []
    section_texts: List[str] = []
    section_sources: List[str] = []
    section_meta: List[Dict[str, Any]] = []
    page_to_chunk_ids: Dict[int, set[int]] = {}
    current_page = 1
    heading_stack: List[tuple[int, str]] = []

    for section in sections:
        current_level = section.get("level", 1)
        chapter_num = section.get("chapter", 0)

        while heading_stack and heading_stack[-1][0] >= current_level:
            heading_stack.pop()

        if section["heading"] != "Introduction":
            heading_stack.append((current_level, section["heading"]))

        full_section_path = " ".join(title for _, title in heading_stack)
        full_section_path = f"Chapter {chapter_num} {full_section_path}".strip()

        sub_chunks = chunker.chunk(section["content"])
        retained_chunk_ids: List[int] = []
        section_pages: set[int] = set()
        cleaned_section_content = _clean_page_markers(section["content"])

        for sub_chunk in sub_chunks:
            chunk_pages, current_page = _extract_chunk_pages(sub_chunk, current_page)
            clean_chunk = _clean_page_markers(sub_chunk)

            if section["heading"] == "Introduction" or not clean_chunk:
                continue

            chunk_id = len(chunk_texts)
            contextualized_chunk = _contextualize_text(full_section_path, chunk_pages, clean_chunk)

            chunk_texts.append(contextualized_chunk)
            chunk_sources.append(str(source_path))
            chunk_meta.append(
                {
                    "filename": str(source_path),
                    "mode": chunk_config.to_string(),
                    "char_len": len(clean_chunk),
                    "word_len": len(clean_chunk.split()),
                    "section": section["heading"],
                    "section_path": full_section_path,
                    "text_preview": clean_chunk[:100],
                    "page_numbers": chunk_pages,
                    "page_span_width": max(1, len(chunk_pages)),
                    "chunk_id": chunk_id,
                    "raw_text": clean_chunk,
                }
            )

            retained_chunk_ids.append(chunk_id)
            section_pages.update(chunk_pages)
            for page in chunk_pages:
                page_to_chunk_ids.setdefault(page, set()).add(chunk_id)

        if not retained_chunk_ids:
            continue

        section_id = len(section_texts)
        section_page_numbers = sorted(section_pages)
        section_texts.append(
            _contextualize_text(full_section_path, section_page_numbers, cleaned_section_content)
        )
        section_sources.append(str(source_path))
        section_meta.append(
            {
                "section_id": section_id,
                "filename": str(source_path),
                "heading": section["heading"],
                "section_path": full_section_path,
                "chapter": chapter_num,
                "chunk_ids": retained_chunk_ids,
                "page_numbers": section_page_numbers,
                "text_preview": cleaned_section_content[:140],
            }
        )

        for chunk_id in retained_chunk_ids:
            chunk_meta[chunk_id]["section_id"] = section_id

    return (chunk_texts, chunk_sources, chunk_meta,
            section_texts, section_sources, section_meta,
            page_to_chunk_ids)


def _embed_and_build_indexes(
    embedding_model_path: str,
    chunk_texts: Sequence[str],
    section_texts: Sequence[str],
    use_multiprocessing: bool,
) -> tuple[np.ndarray, np.ndarray, faiss.Index, faiss.Index, BM25Okapi, BM25Okapi]:
    """Embed chunk and section texts, then build FAISS and BM25 indexes for each.

    Returns (chunk_embeddings, section_embeddings,
    chunk_faiss, section_faiss, chunk_bm25, section_bm25).
    """
    print(f"Embedding {len(chunk_texts):,} chunks with {pathlib.Path(embedding_model_path).stem} ...")
    embedder = SentenceTransformer(embedding_model_path)

    chunk_embeddings = _embed_texts(
        embedder=embedder, texts=chunk_texts,
        use_multiprocessing=use_multiprocessing,
        batch_size=32 if use_multiprocessing else 8,
    )
    print(f"Embedding {len(section_texts):,} sections...")
    section_embeddings = _embed_texts(
        embedder=embedder, texts=section_texts,
        use_multiprocessing=use_multiprocessing,
        batch_size=16 if use_multiprocessing else 8,
    )

    print(f"Building FAISS + BM25 indexes for {len(chunk_texts):,} chunks and {len(section_texts):,} sections...")
    chunk_faiss = _build_faiss_index(chunk_embeddings)
    section_faiss = _build_faiss_index(section_embeddings)
    chunk_bm25 = _build_bm25_index(chunk_texts)
    section_bm25 = _build_bm25_index(section_texts)

    return (chunk_embeddings, section_embeddings,
            chunk_faiss, section_faiss, chunk_bm25, section_bm25)


def _persist_artifacts(
    *,
    artifacts_dir: pathlib.Path,
    file_map: Dict[str, str],
    index_prefix: str,
    markdown_file: str,
    embedding_model_path: str,
    chunk_texts: Sequence[str],
    chunk_sources: Sequence[str],
    chunk_meta: Sequence[Dict[str, Any]],
    section_texts: Sequence[str],
    section_sources: Sequence[str],
    section_meta: Sequence[Dict[str, Any]],
    page_to_chunk_ids: Dict[int, set[int]],
    chunk_embeddings: np.ndarray,
    section_embeddings: np.ndarray,
    chunk_faiss: faiss.Index,
    section_faiss: faiss.Index,
    chunk_bm25: BM25Okapi,
    section_bm25: BM25Okapi,
    chunk_config: ChunkConfig,
    use_headings: bool,
    use_multiprocessing: bool,
) -> None:
    """Write all index artifacts, page maps, and manifest to disk."""
    final_page_map = {page: sorted(ids) for page, ids in page_to_chunk_ids.items()}
    with (artifacts_dir / file_map["page_to_chunk_map"]).open("w", encoding="utf-8") as handle:
        json.dump(final_page_map, handle, indent=2)

    np.save(artifacts_dir / file_map["chunk_embeddings"], chunk_embeddings)
    np.save(artifacts_dir / file_map["section_embeddings"], section_embeddings)

    faiss.write_index(chunk_faiss, str(artifacts_dir / file_map["chunk_index"]))
    faiss.write_index(section_faiss, str(artifacts_dir / file_map["section_index"]))

    for key, data in [
        ("chunk_bm25", chunk_bm25), ("section_bm25", section_bm25),
        ("chunks", list(chunk_texts)), ("sources", list(chunk_sources)),
        ("metadata", list(chunk_meta)),
        ("sections", list(section_texts)), ("section_sources", list(section_sources)),
        ("section_meta", list(section_meta)),
    ]:
        with (artifacts_dir / file_map[key]).open("wb") as handle:
            pickle.dump(data, handle)

    manifest = build_manifest(
        index_prefix=index_prefix,
        source_document=markdown_file,
        embedding_model_path=embedding_model_path,
        chunk_count=len(chunk_texts),
        section_count=len(section_texts),
        file_map=file_map,
        build_settings={
            "chunk_mode": chunk_config.to_string(),
            "use_headings": bool(use_headings),
            "use_multiprocessing": bool(use_multiprocessing),
        },
    )
    manifest_path = save_manifest(
        artifacts_dir=artifacts_dir,
        index_prefix=index_prefix,
        manifest=manifest,
    )
    print(f"Saved all index artifacts with prefix: {index_prefix}")
    print(f"Saved artifact manifest: {manifest_path}")


def build_index(
    markdown_file: str,
    *,
    chunker: DocumentChunker,
    chunk_config: ChunkConfig,
    embedding_model_path: str,
    embedding_model_context_window: int,
    artifacts_dir: os.PathLike,
    index_prefix: str,
    use_multiprocessing: bool = False,
    use_headings: bool = False,
) -> None:
    """Extract sections, chunk, embed, and build both section-level and chunk-level indexes."""
    artifacts_path = pathlib.Path(artifacts_dir)
    artifacts_path.mkdir(parents=True, exist_ok=True)
    file_map = artifact_file_map(index_prefix)

    (chunk_texts, chunk_sources, chunk_meta,
     section_texts, section_sources, section_meta,
     page_to_chunk_ids) = _extract_and_chunk(markdown_file, chunker, chunk_config)

    (chunk_embeddings, section_embeddings,
     chunk_faiss, section_faiss,
     chunk_bm25, section_bm25) = _embed_and_build_indexes(
        embedding_model_path, chunk_texts, section_texts, use_multiprocessing,
    )

    _persist_artifacts(
        artifacts_dir=artifacts_path,
        file_map=file_map,
        index_prefix=index_prefix,
        markdown_file=markdown_file,
        embedding_model_path=embedding_model_path,
        chunk_texts=chunk_texts,
        chunk_sources=chunk_sources,
        chunk_meta=chunk_meta,
        section_texts=section_texts,
        section_sources=section_sources,
        section_meta=section_meta,
        page_to_chunk_ids=page_to_chunk_ids,
        chunk_embeddings=chunk_embeddings,
        section_embeddings=section_embeddings,
        chunk_faiss=chunk_faiss,
        section_faiss=section_faiss,
        chunk_bm25=chunk_bm25,
        section_bm25=section_bm25,
        chunk_config=chunk_config,
        use_headings=use_headings,
        use_multiprocessing=use_multiprocessing,
    )


def preprocess_for_bm25(text: str) -> List[str]:
    """
    Simplify text for BM25 tokenization.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9_'#+-]", " ", text)
    return text.split()
