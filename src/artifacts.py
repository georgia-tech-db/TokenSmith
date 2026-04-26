from __future__ import annotations

import hashlib
import json
import pathlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import faiss
import numpy as np


ARTIFACT_VERSION = 2


class ArtifactValidationError(RuntimeError):
    """Raised when persisted index artifacts are missing or inconsistent."""


@dataclass
class ArtifactBundle:
    """Container for all persisted index artifacts used during retrieval."""

    chunk_index: faiss.Index
    chunk_bm25: Any
    chunks: List[str]
    sources: List[str]
    metadata: List[Dict[str, Any]]
    page_to_chunk_map: Dict[int, List[int]] = field(default_factory=dict)
    chunk_embeddings: Optional[np.ndarray] = None
    section_index: Optional[faiss.Index] = None
    section_bm25: Any = None
    sections: List[str] = field(default_factory=list)
    section_sources: List[str] = field(default_factory=list)
    section_meta: List[Dict[str, Any]] = field(default_factory=list)
    section_embeddings: Optional[np.ndarray] = None
    manifest: Optional[Dict[str, Any]] = None

    @property
    def has_hierarchical_artifacts(self) -> bool:
        return bool(self.sections and self.section_meta and self.section_index is not None)


def artifact_file_map(index_prefix: str) -> Dict[str, str]:
    """Return a mapping of artifact logical names to their file paths for a given index prefix."""
    return {
        "chunk_index": f"{index_prefix}.faiss",
        "chunk_bm25": f"{index_prefix}_bm25.pkl",
        "chunks": f"{index_prefix}_chunks.pkl",
        "sources": f"{index_prefix}_sources.pkl",
        "metadata": f"{index_prefix}_meta.pkl",
        "page_to_chunk_map": f"{index_prefix}_page_to_chunk_map.json",
        "chunk_embeddings": f"{index_prefix}_chunk_embeddings.npy",
        "section_index": f"{index_prefix}_sections.faiss",
        "section_bm25": f"{index_prefix}_sections_bm25.pkl",
        "sections": f"{index_prefix}_sections.pkl",
        "section_sources": f"{index_prefix}_section_sources.pkl",
        "section_meta": f"{index_prefix}_section_meta.pkl",
        "section_embeddings": f"{index_prefix}_section_embeddings.npy",
        "manifest": f"{index_prefix}_manifest.json",
    }


def sha256_file(path: pathlib.Path) -> str:
    """Compute the SHA-256 hex digest of a file, reading in 1 MB blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_manifest(
    *,
    index_prefix: str,
    source_document: str,
    embedding_model_path: str,
    chunk_count: int,
    section_count: int,
    file_map: Dict[str, str],
    build_settings: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a manifest dict capturing artifact metadata, source hash, and build settings.

    Args:
        index_prefix: Prefix used to locate artifact files on disk.
        source_document: Path to the original source document.
        embedding_model_path: Path or name of the embedding model used.
        chunk_count: Number of chunks in the index.
        section_count: Number of sections in the index.
        file_map: Mapping of artifact names to relative file paths.
        build_settings: Optional dict of index-build configuration.

    Returns:
        A manifest dictionary suitable for JSON serialization.
    """
    source_path = pathlib.Path(source_document)
    return {
        "artifact_version": ARTIFACT_VERSION,
        "index_prefix": index_prefix,
        "source_document": str(source_path),
        "source_document_sha256": sha256_file(source_path),
        "embedding_model_path": embedding_model_path,
        "chunk_count": chunk_count,
        "section_count": section_count,
        "files": file_map,
        "build_settings": build_settings or {},
    }


def save_manifest(
    *,
    artifacts_dir: pathlib.Path,
    index_prefix: str,
    manifest: Dict[str, Any],
) -> pathlib.Path:
    """Write a manifest dict to its canonical JSON file and return the path."""
    manifest_path = artifacts_dir / artifact_file_map(index_prefix)["manifest"]
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    return manifest_path


def load_manifest(artifacts_dir: pathlib.Path, index_prefix: str) -> Optional[Dict[str, Any]]:
    """Load a manifest from disk, returning None if the file does not exist."""
    manifest_path = artifacts_dir / artifact_file_map(index_prefix)["manifest"]
    if not manifest_path.exists():
        return None
    with manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_bundle(bundle: ArtifactBundle) -> None:
    """Validate chunk/page/section relationships for hierarchical retrieval.

    Checks performed:
        - Chunk, source, and metadata lists have consistent lengths.
        - Chunk embeddings count matches chunk count (if present).
        - Each chunk metadata entry has a sequential ``chunk_id``.
        - Section references in chunk metadata point to existing sections.
        - Page-to-chunk map entries reference valid chunk ids.
        - Section artifacts have consistent lengths (if hierarchical).
        - Section embeddings count matches section count (if present).
        - Bidirectional chunk-section references are consistent.

    Raises:
        ArtifactValidationError: If any consistency check fails.
    """
    chunk_count = len(bundle.chunks)
    if chunk_count != len(bundle.sources) or chunk_count != len(bundle.metadata):
        raise ArtifactValidationError(
            "Chunk artifacts disagree on count: "
            f"chunks={len(bundle.chunks)} sources={len(bundle.sources)} meta={len(bundle.metadata)}"
        )

    if bundle.chunk_embeddings is not None and len(bundle.chunk_embeddings) != chunk_count:
        raise ArtifactValidationError(
            "Chunk embeddings disagree with chunk count: "
            f"embeddings={len(bundle.chunk_embeddings)} chunks={chunk_count}"
        )

    for expected_chunk_id, meta in enumerate(bundle.metadata):
        chunk_id = meta.get("chunk_id")
        if chunk_id != expected_chunk_id:
            raise ArtifactValidationError(
                f"Chunk metadata out of sync at position {expected_chunk_id}: stored chunk_id={chunk_id}"
            )

        section_id = meta.get("section_id")
        if section_id is not None and section_id >= len(bundle.section_meta):
            raise ArtifactValidationError(
                f"Chunk {expected_chunk_id} references missing section {section_id}"
            )

    for raw_page, chunk_ids in bundle.page_to_chunk_map.items():
        for chunk_id in chunk_ids:
            if chunk_id < 0 or chunk_id >= chunk_count:
                raise ArtifactValidationError(
                    f"Page {raw_page} references invalid chunk id {chunk_id}"
                )

    if not bundle.has_hierarchical_artifacts:
        return

    section_count = len(bundle.sections)
    if section_count != len(bundle.section_sources) or section_count != len(bundle.section_meta):
        raise ArtifactValidationError(
            "Section artifacts disagree on count: "
            f"sections={len(bundle.sections)} sources={len(bundle.section_sources)} meta={len(bundle.section_meta)}"
        )

    if bundle.section_embeddings is not None and len(bundle.section_embeddings) != section_count:
        raise ArtifactValidationError(
            "Section embeddings disagree with section count: "
            f"embeddings={len(bundle.section_embeddings)} sections={section_count}"
        )

    for section_id, section_meta in enumerate(bundle.section_meta):
        if section_meta.get("section_id") != section_id:
            raise ArtifactValidationError(
                f"Section metadata out of sync at position {section_id}: "
                f"stored section_id={section_meta.get('section_id')}"
            )

        chunk_ids = section_meta.get("chunk_ids", [])
        for chunk_id in chunk_ids:
            if chunk_id < 0 or chunk_id >= chunk_count:
                raise ArtifactValidationError(
                    f"Section {section_id} references invalid chunk id {chunk_id}"
                )
            if bundle.metadata[chunk_id].get("section_id") != section_id:
                raise ArtifactValidationError(
                    f"Chunk {chunk_id} does not point back to section {section_id}"
                )
