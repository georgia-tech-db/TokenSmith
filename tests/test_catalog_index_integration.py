"""Integration checks for index catalog wiring (no full embedding run)."""

import argparse
from pathlib import Path

from src.config import RAGConfig
from src.index_cli import run_index_mode


def test_run_index_mode_forwards_catalog_db_path_to_build_index(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "stub.md").write_text("# Section\n\nBody.\n")

    calls = []

    def fake_build_index(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})

    cfg = RAGConfig()
    cfg.catalog_db_path = str(tmp_path / "catalog.db")

    args = argparse.Namespace(
        index_prefix="textbook_index",
        keep_tables=False,
        multiproc_indexing=False,
        embed_with_headings=False,
    )
    run_index_mode(args, cfg, build_index_fn=fake_build_index)

    assert len(calls) == 1
    assert calls[0]["kwargs"]["catalog_db_path"] == str(tmp_path / "catalog.db")
    assert calls[0]["kwargs"]["index_prefix"] == "textbook_index"
    assert Path(calls[0]["kwargs"]["markdown_file"]).name == "stub.md"


def test_run_index_mode_passes_none_catalog_when_unset(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "stub.md").write_text("# Section\n\nBody.\n")

    calls = []

    def fake_build_index(*args, **kwargs):
        calls.append(kwargs)

    cfg = RAGConfig()
    cfg.catalog_db_path = None

    args = argparse.Namespace(
        index_prefix="pfx",
        keep_tables=False,
        multiproc_indexing=False,
        embed_with_headings=False,
    )
    run_index_mode(args, cfg, build_index_fn=fake_build_index)

    assert calls[0]["catalog_db_path"] is None
