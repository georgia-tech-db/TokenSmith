#!/usr/bin/env python3
"""
index_updater.py
Maintains partial chapter indexes using the same artifact contract as full indexing.
"""

from __future__ import annotations

import json
import os
import pathlib
from typing import List, Union

from src.artifacts import ArtifactValidationError
from src.index_builder import build_index
from src.preprocessing.chunking import ChunkConfig, DocumentChunker
from src.retriever import load_artifact_bundle


def _load_existing_chapters(info_path: pathlib.Path, markdown_file: str) -> List[Union[int, str]]:
    """Return chapters already recorded for a source document in index info."""
    if not info_path.exists():
        return []

    with info_path.open("r", encoding="utf-8") as handle:
        index_info = json.load(handle)

    for textbook in index_info.get("textbooks", []):
        if textbook.get("markdown_file") != markdown_file:
            continue
        chapters = textbook.get("chapters", [])
        if "all" in chapters:
            return ["all"]
        return [int(chapter) for chapter in chapters]
    return []


def add_to_index(
    markdown_file: str,
    *,
    chunker: DocumentChunker,
    chunk_config: ChunkConfig,
    embedding_model_path: str,
    embedding_model_context_window: int,
    artifacts_dir: os.PathLike,
    index_prefix: str,
    chapters_to_add: List[int],
    use_multiprocessing: bool = False,
    use_headings: bool = False,
) -> None:
    """
    Add chapters to a partial index by rebuilding all artifacts atomically.

    Reusing ``build_index`` keeps chunk, section, embedding, BM25, FAISS, page-map,
    and manifest artifacts in one consistent format. This is intentionally more
    conservative than appending only chunk-level files.
    """
    artifacts_path = pathlib.Path(artifacts_dir)
    artifacts_path.mkdir(parents=True, exist_ok=True)
    info_path = artifacts_path / f"{index_prefix}_info.json"

    existing_chapters = _load_existing_chapters(info_path, markdown_file)
    if "all" in existing_chapters:
        print("Index contains all chapters. No new chapters to add.")
        return

    merged_chapters = sorted({int(chapter) for chapter in existing_chapters + chapters_to_add})
    if not merged_chapters:
        print("No chapters requested.")
        return

    if set(merged_chapters) == set(existing_chapters):
        print("All requested chapters are already in the index.")
        return

    print(f"Rebuilding partial index for chapters {merged_chapters}...")
    build_index(
        markdown_file=markdown_file,
        chunker=chunker,
        chunk_config=chunk_config,
        embedding_model_path=embedding_model_path,
        embedding_model_context_window=embedding_model_context_window,
        artifacts_dir=artifacts_path,
        index_prefix=index_prefix,
        use_multiprocessing=use_multiprocessing,
        use_headings=use_headings,
        chapters_to_index=merged_chapters,
    )

    try:
        load_artifact_bundle(artifacts_path, index_prefix)
    except ArtifactValidationError as exc:
        raise RuntimeError("Rebuilt index artifacts failed validation") from exc
