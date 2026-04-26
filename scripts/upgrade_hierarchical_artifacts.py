#!/usr/bin/env python3
"""Upgrade existing chunk-only TokenSmith artifacts to hierarchical artifacts.

This script is a low-memory workaround for environments where rebuilding the full
chunk index is too expensive. It:

1. Loads the existing chunk-level artifacts and FAISS index.
2. Reconstructs chunk embeddings from the stored FAISS vectors.
3. Adds section ids and raw-text fields to chunk metadata.
4. Builds section texts, section embeddings, section BM25, section FAISS, and a manifest.

The script assumes the existing chunk artifacts already contain contextualized
chunk text and a ``section_path`` field in metadata.
"""

from __future__ import annotations

import argparse
import pathlib
import pickle
import re
from typing import Any, Dict, List

import faiss
import numpy as np

from src.artifacts import artifact_file_map, build_manifest, save_manifest
from src.embedder import SentenceTransformer
from src.index_builder import _build_bm25_index, _contextualize_text
from src.retriever import load_artifact_bundle


OLD_CONTEXT_RE = re.compile(r"^Description:\s*(?P<section>.*?)\s+Content:\s*(?P<body>.*)$", re.DOTALL)
NEW_CONTEXT_RE = re.compile(
    r"^Section Path:\s*(?P<section>.*?)\nPage Span:\s*.*?\nContent:\n(?P<body>.*)$",
    re.DOTALL,
)
CHAPTER_RE = re.compile(r"Chapter\s+(?P<chapter>\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upgrade TokenSmith artifacts to hierarchical form.")
    parser.add_argument(
        "--artifacts-dir",
        default="index/sections",
        help="Directory containing persisted TokenSmith artifacts.",
    )
    parser.add_argument(
        "--index-prefix",
        default="textbook_index",
        help="Artifact prefix used inside the artifacts directory.",
    )
    parser.add_argument(
        "--embed-model",
        default="models/embedders/Qwen3-Embedding-4B-Q5_K_M.gguf",
        help="Embedding model path recorded in the manifest.",
    )
    parser.add_argument(
        "--source-document",
        default=None,
        help="Optional source document path for the manifest. Defaults to the first chunk source.",
    )
    parser.add_argument(
        "--embedding-context",
        type=int,
        default=1024,
        help="Context window to use when loading the section embedding model.",
    )
    parser.add_argument(
        "--max-section-chars",
        type=int,
        default=6000,
        help="Maximum number of characters kept per section when building section texts.",
    )
    parser.add_argument(
        "--section-embedding-mode",
        choices=["mean_chunk", "llama"],
        default="mean_chunk",
        help="How to derive section embeddings.",
    )
    return parser.parse_args()


def _extract_raw_text(chunk_text: str, fallback_section_path: str) -> str:
    """Strip contextual prefixes from a stored chunk text."""
    for pattern in (NEW_CONTEXT_RE, OLD_CONTEXT_RE):
        match = pattern.match(chunk_text)
        if match:
            return match.group("body").strip()

    prefix = fallback_section_path.strip()
    if chunk_text.startswith(prefix):
        return chunk_text[len(prefix):].strip()
    return chunk_text.strip()


def _derive_heading(section_path: str) -> str:
    """Extract a human-readable heading from a section path."""
    if " Section " in section_path:
        return section_path.rsplit(" Section ", 1)[-1].strip()
    return section_path.strip()


def _derive_chapter(section_path: str) -> int:
    """Extract the leading chapter number from a section path, defaulting to 0."""
    match = CHAPTER_RE.search(section_path)
    if match:
        return int(match.group("chapter"))
    return 0


def _reconstruct_chunk_embeddings(index: faiss.Index, expected_count: int) -> np.ndarray:
    """Reconstruct stored chunk embeddings from the persisted FAISS index."""
    if index.ntotal != expected_count:
        raise ValueError(
            f"Chunk index count mismatch: index has {index.ntotal}, metadata has {expected_count}"
        )
    return np.vstack(
        [np.asarray(index.reconstruct(i), dtype=np.float32) for i in range(expected_count)],
        dtype=np.float32,
    )


def _persist_pickle(path: pathlib.Path, payload: Any) -> None:
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def build_hierarchical_artifacts(
    *,
    artifacts_dir: pathlib.Path,
    index_prefix: str,
    embed_model: str,
    source_document: str | None,
    embedding_context: int,
    max_section_chars: int,
    section_embedding_mode: str,
) -> None:
    """Upgrade chunk-only artifacts in-place to hierarchical artifacts."""
    bundle = load_artifact_bundle(artifacts_dir, index_prefix)
    file_map = artifact_file_map(index_prefix)

    updated_metadata: List[Dict[str, Any]] = []
    section_order: List[str] = []
    section_groups: Dict[str, Dict[str, Any]] = {}

    for chunk_id, (chunk_text, source, meta) in enumerate(
        zip(bundle.chunks, bundle.sources, bundle.metadata, strict=True)
    ):
        section_path = meta.get("section_path")
        if not section_path:
            raise ValueError(f"Chunk {chunk_id} is missing section_path metadata")

        raw_text = meta.get("raw_text") or _extract_raw_text(chunk_text, section_path)
        page_numbers = sorted({int(page) for page in meta.get("page_numbers", [])})

        if section_path not in section_groups:
            section_order.append(section_path)
            section_groups[section_path] = {
                "filename": source,
                "chunk_ids": [],
                "page_numbers": set(),
                "raw_texts": [],
            }

        group = section_groups[section_path]
        group["chunk_ids"].append(chunk_id)
        group["page_numbers"].update(page_numbers)
        group["raw_texts"].append(raw_text)

        updated_meta = dict(meta)
        updated_meta["chunk_id"] = chunk_id
        updated_meta["raw_text"] = raw_text
        updated_meta["page_span_width"] = max(1, len(page_numbers))
        updated_metadata.append(updated_meta)

    section_texts: List[str] = []
    section_sources: List[str] = []
    section_meta: List[Dict[str, Any]] = []

    for section_id, section_path in enumerate(section_order):
        group = section_groups[section_path]
        page_numbers = sorted(int(page) for page in group["page_numbers"])
        combined_text = "\n\n".join(text.strip() for text in group["raw_texts"] if text.strip())
        section_text = _contextualize_text(
            section_path,
            page_numbers,
            combined_text[:max_section_chars],
        )
        section_texts.append(section_text)
        section_sources.append(str(group["filename"]))
        section_meta.append(
            {
                "section_id": section_id,
                "filename": str(group["filename"]),
                "heading": _derive_heading(section_path),
                "section_path": section_path,
                "chapter": _derive_chapter(section_path),
                "chunk_ids": list(group["chunk_ids"]),
                "page_numbers": page_numbers,
                "text_preview": combined_text[:140],
            }
        )
        for chunk_id in group["chunk_ids"]:
            updated_metadata[chunk_id]["section_id"] = section_id

    print(f"Reconstructing {len(updated_metadata):,} chunk embeddings from FAISS...")
    chunk_embeddings = _reconstruct_chunk_embeddings(bundle.chunk_index, len(updated_metadata))

    if section_embedding_mode == "mean_chunk":
        print(f"Deriving {len(section_texts):,} section embeddings by averaging chunk vectors...")
        section_embeddings = np.vstack(
            [
                np.mean(chunk_embeddings[section["chunk_ids"]], axis=0, dtype=np.float32)
                for section in section_meta
            ]
        ).astype(np.float32)
    else:
        print(f"Embedding {len(section_texts):,} sections with {pathlib.Path(embed_model).stem} ...")
        embedder = SentenceTransformer(embed_model, n_ctx=embedding_context)
        section_embeddings = embedder.encode(
            section_texts,
            batch_size=8,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

    print("Building section-level BM25 and FAISS indexes...")
    section_index = faiss.IndexFlatL2(section_embeddings.shape[1])
    section_index.add(section_embeddings)
    section_bm25 = _build_bm25_index(section_texts)

    print("Persisting upgraded artifacts...")
    np.save(artifacts_dir / file_map["chunk_embeddings"], chunk_embeddings)
    np.save(artifacts_dir / file_map["section_embeddings"], section_embeddings)
    faiss.write_index(section_index, str(artifacts_dir / file_map["section_index"]))

    _persist_pickle(artifacts_dir / file_map["metadata"], updated_metadata)
    _persist_pickle(artifacts_dir / file_map["sections"], section_texts)
    _persist_pickle(artifacts_dir / file_map["section_sources"], section_sources)
    _persist_pickle(artifacts_dir / file_map["section_meta"], section_meta)
    _persist_pickle(artifacts_dir / file_map["section_bm25"], section_bm25)

    manifest = build_manifest(
        index_prefix=index_prefix,
        source_document=source_document or bundle.sources[0],
        embedding_model_path=embed_model,
        chunk_count=len(bundle.chunks),
        section_count=len(section_texts),
        file_map=file_map,
        build_settings={
            "artifact_upgrade": "hierarchical_from_existing_chunks",
            "section_embedding_mode": section_embedding_mode,
            "chunk_mode": updated_metadata[0].get("mode"),
            "use_headings": True,
            "use_multiprocessing": False,
        },
    )
    manifest_path = save_manifest(
        artifacts_dir=artifacts_dir,
        index_prefix=index_prefix,
        manifest=manifest,
    )
    print(f"Saved hierarchical artifacts manifest to {manifest_path}")


def main() -> None:
    args = parse_args()
    build_hierarchical_artifacts(
        artifacts_dir=pathlib.Path(args.artifacts_dir),
        index_prefix=args.index_prefix,
        embed_model=args.embed_model,
        source_document=args.source_document,
        embedding_context=args.embedding_context,
        max_section_chars=args.max_section_chars,
        section_embedding_mode=args.section_embedding_mode,
    )


if __name__ == "__main__":
    main()
