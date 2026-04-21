import pathlib
import pickle

from src.catalog.index_catalog import IndexCatalog
from src.retriever import load_artifacts


class _FakeFaissIndex:
    pass


class _FakeBm25:
    pass


def test_index_catalog_write_and_load_latest(tmp_path):
    db = tmp_path / "catalog.db"
    catalog = IndexCatalog(db)
    chunks = ["c0", "c1"]
    sources = ["doc.md", "doc.md"]
    metadata = [
        {"chunk_id": 0, "page_numbers": [1], "section": "A"},
        {"chunk_id": 1, "page_numbers": [2], "section": "B"},
    ]
    b1 = catalog.write_build(
        index_prefix="textbook_index",
        artifacts_dir=tmp_path / "index",
        source_markdown="data/doc.md",
        chunks=chunks,
        sources=sources,
        metadata=metadata,
    )
    b2 = catalog.write_build(
        index_prefix="textbook_index",
        artifacts_dir=tmp_path / "index",
        source_markdown="data/doc.md",
        chunks=["new"],
        sources=["doc.md"],
        metadata=[{"chunk_id": 0, "page_numbers": [9], "section": "new"}],
    )
    assert b2 > b1

    loaded = catalog.load_latest_build(index_prefix="textbook_index")
    assert loaded is not None
    loaded_chunks, loaded_sources, loaded_meta = loaded
    assert loaded_chunks == ["new"]
    assert loaded_sources == ["doc.md"]
    assert loaded_meta[0]["page_numbers"] == [9]


def test_load_artifacts_prefers_catalog_with_pickle_fallback(tmp_path, monkeypatch):
    artifacts_dir = tmp_path / "index" / "sections"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    prefix = "textbook_index"
    # Only bm25/faiss are always read from files.
    (artifacts_dir / f"{prefix}.faiss").write_bytes(b"FAKE")
    with open(artifacts_dir / f"{prefix}_bm25.pkl", "wb") as f:
        pickle.dump(_FakeBm25(), f)
    with open(artifacts_dir / f"{prefix}_chunks.pkl", "wb") as f:
        pickle.dump(["pickle_chunk"], f)
    with open(artifacts_dir / f"{prefix}_sources.pkl", "wb") as f:
        pickle.dump(["pickle.md"], f)
    with open(artifacts_dir / f"{prefix}_meta.pkl", "wb") as f:
        pickle.dump([{"chunk_id": 0, "page_numbers": [5]}], f)

    # Patch faiss.read_index so we can run this unit test without real FAISS files.
    import src.retriever as retriever_mod

    monkeypatch.setattr(retriever_mod.faiss, "read_index", lambda _: _FakeFaissIndex())

    catalog_db = tmp_path / "catalog.db"
    catalog = IndexCatalog(catalog_db)
    catalog.write_build(
        index_prefix=prefix,
        artifacts_dir=artifacts_dir,
        source_markdown="data/doc.md",
        chunks=["catalog_chunk"],
        sources=["catalog.md"],
        metadata=[{"chunk_id": 0, "page_numbers": [7]}],
    )

    _, _, chunks, sources, metadata = load_artifacts(
        artifacts_dir=artifacts_dir,
        index_prefix=prefix,
        catalog_db_path=catalog_db,
    )
    assert chunks == ["catalog_chunk"]
    assert sources == ["catalog.md"]
    assert metadata[0]["page_numbers"] == [7]

    # If catalog path is unset, fallback to pickle.
    _, _, chunks2, sources2, metadata2 = load_artifacts(
        artifacts_dir=artifacts_dir,
        index_prefix=prefix,
    )
    assert chunks2 == ["pickle_chunk"]
    assert sources2 == ["pickle.md"]
    assert metadata2[0]["page_numbers"] == [5]
