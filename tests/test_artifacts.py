import json
import pickle

import faiss
import numpy as np
import pytest

from src.artifacts import (
    ARTIFACT_VERSION,
    ArtifactBundle,
    ArtifactValidationError,
    artifact_file_map,
    build_manifest,
    save_manifest,
    validate_bundle,
)
from src.retriever import load_artifact_bundle


pytestmark = pytest.mark.unit


def make_bundle():
    chunk_embeddings = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    section_embeddings = np.array([[0.5, 0.5]], dtype=np.float32)

    chunk_index = faiss.IndexFlatL2(2)
    chunk_index.add(chunk_embeddings)
    section_index = faiss.IndexFlatL2(2)
    section_index.add(section_embeddings)

    return ArtifactBundle(
        chunk_index=chunk_index,
        chunk_bm25=object(),
        chunks=["c0", "c1"],
        sources=["doc", "doc"],
        metadata=[
            {"chunk_id": 0, "section_id": 0, "page_numbers": [1]},
            {"chunk_id": 1, "section_id": 0, "page_numbers": [2]},
        ],
        page_to_chunk_map={1: [0], 2: [1]},
        chunk_embeddings=chunk_embeddings,
        section_index=section_index,
        section_bm25=object(),
        sections=["section"],
        section_sources=["doc"],
        section_meta=[{"section_id": 0, "chunk_ids": [0, 1], "page_numbers": [1, 2]}],
        section_embeddings=section_embeddings,
        manifest={"artifact_version": 2},
    )


def test_validate_bundle_accepts_consistent_bundle():
    validate_bundle(make_bundle())


def test_validate_bundle_rejects_chunk_id_mismatch():
    bundle = make_bundle()
    bundle.metadata[1]["chunk_id"] = 7
    with pytest.raises(ArtifactValidationError):
        validate_bundle(bundle)


def test_validate_bundle_rejects_broken_section_backpointer():
    bundle = make_bundle()
    bundle.section_meta[0]["chunk_ids"] = [0, 9]
    with pytest.raises(ArtifactValidationError):
        validate_bundle(bundle)


def test_load_artifact_bundle_rejects_source_hash_mismatch(tmp_path):
    source_doc = tmp_path / "textbook.json"
    source_doc.write_text('[{"heading":"Section 1.1 A","content":"x","level":2,"chapter":1}]', encoding="utf-8")
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    file_map = artifact_file_map("textbook_index")

    chunk_embeddings = np.array([[0.0, 0.0]], dtype=np.float32)
    section_embeddings = np.array([[0.0, 0.0]], dtype=np.float32)

    chunk_index = faiss.IndexFlatL2(2)
    chunk_index.add(chunk_embeddings)
    section_index = faiss.IndexFlatL2(2)
    section_index.add(section_embeddings)

    faiss.write_index(chunk_index, str(artifacts_dir / file_map["chunk_index"]))
    faiss.write_index(section_index, str(artifacts_dir / file_map["section_index"]))
    np.save(artifacts_dir / file_map["chunk_embeddings"], chunk_embeddings)
    np.save(artifacts_dir / file_map["section_embeddings"], section_embeddings)

    with (artifacts_dir / file_map["chunk_bm25"]).open("wb") as handle:
        pickle.dump({"bm25": "chunk"}, handle)
    with (artifacts_dir / file_map["section_bm25"]).open("wb") as handle:
        pickle.dump({"bm25": "section"}, handle)
    with (artifacts_dir / file_map["chunks"]).open("wb") as handle:
        pickle.dump(["chunk"], handle)
    with (artifacts_dir / file_map["sources"]).open("wb") as handle:
        pickle.dump([str(source_doc)], handle)
    with (artifacts_dir / file_map["metadata"]).open("wb") as handle:
        pickle.dump([{"chunk_id": 0, "section_id": 0, "page_numbers": [1]}], handle)
    with (artifacts_dir / file_map["sections"]).open("wb") as handle:
        pickle.dump(["section"], handle)
    with (artifacts_dir / file_map["section_sources"]).open("wb") as handle:
        pickle.dump([str(source_doc)], handle)
    with (artifacts_dir / file_map["section_meta"]).open("wb") as handle:
        pickle.dump([{"section_id": 0, "chunk_ids": [0], "page_numbers": [1]}], handle)
    with (artifacts_dir / file_map["page_to_chunk_map"]).open("w", encoding="utf-8") as handle:
        json.dump({"1": [0]}, handle)

    manifest = build_manifest(
        index_prefix="textbook_index",
        source_document=str(source_doc),
        embedding_model_path="models/embed.gguf",
        chunk_count=1,
        section_count=1,
        file_map=file_map,
    )
    save_manifest(artifacts_dir=artifacts_dir, index_prefix="textbook_index", manifest=manifest)

    source_doc.write_text('[{"heading":"Section 1.1 A","content":"changed","level":2,"chapter":1}]', encoding="utf-8")

    with pytest.raises(ArtifactValidationError):
        load_artifact_bundle(artifacts_dir, "textbook_index")


def test_load_artifact_bundle_rejects_unsupported_manifest_version(tmp_path):
    source_doc = tmp_path / "textbook.json"
    source_doc.write_text('[{"heading":"Section 1.1 A","content":"x","level":2,"chapter":1}]', encoding="utf-8")
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    file_map = artifact_file_map("textbook_index")

    chunk_embeddings = np.array([[0.0, 0.0]], dtype=np.float32)
    section_embeddings = np.array([[0.0, 0.0]], dtype=np.float32)

    chunk_index = faiss.IndexFlatL2(2)
    chunk_index.add(chunk_embeddings)
    section_index = faiss.IndexFlatL2(2)
    section_index.add(section_embeddings)

    faiss.write_index(chunk_index, str(artifacts_dir / file_map["chunk_index"]))
    faiss.write_index(section_index, str(artifacts_dir / file_map["section_index"]))
    np.save(artifacts_dir / file_map["chunk_embeddings"], chunk_embeddings)
    np.save(artifacts_dir / file_map["section_embeddings"], section_embeddings)

    with (artifacts_dir / file_map["chunk_bm25"]).open("wb") as handle:
        pickle.dump({"bm25": "chunk"}, handle)
    with (artifacts_dir / file_map["section_bm25"]).open("wb") as handle:
        pickle.dump({"bm25": "section"}, handle)
    with (artifacts_dir / file_map["chunks"]).open("wb") as handle:
        pickle.dump(["chunk"], handle)
    with (artifacts_dir / file_map["sources"]).open("wb") as handle:
        pickle.dump([str(source_doc)], handle)
    with (artifacts_dir / file_map["metadata"]).open("wb") as handle:
        pickle.dump([{"chunk_id": 0, "section_id": 0, "page_numbers": [1]}], handle)
    with (artifacts_dir / file_map["sections"]).open("wb") as handle:
        pickle.dump(["section"], handle)
    with (artifacts_dir / file_map["section_sources"]).open("wb") as handle:
        pickle.dump([str(source_doc)], handle)
    with (artifacts_dir / file_map["section_meta"]).open("wb") as handle:
        pickle.dump([{"section_id": 0, "chunk_ids": [0], "page_numbers": [1]}], handle)
    with (artifacts_dir / file_map["page_to_chunk_map"]).open("w", encoding="utf-8") as handle:
        json.dump({"1": [0]}, handle)

    manifest = build_manifest(
        index_prefix="textbook_index",
        source_document=str(source_doc),
        embedding_model_path="models/embed.gguf",
        chunk_count=1,
        section_count=1,
        file_map=file_map,
    )
    manifest["artifact_version"] = ARTIFACT_VERSION - 1
    save_manifest(artifacts_dir=artifacts_dir, index_prefix="textbook_index", manifest=manifest)

    with pytest.raises(ArtifactValidationError, match="Unsupported artifact manifest version"):
        load_artifact_bundle(artifacts_dir, "textbook_index")
