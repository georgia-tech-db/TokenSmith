"""Tests for SQLite workload catalog (queries + retrieval_hits)."""

import json
import sqlite3
from pathlib import Path

import pytest

from src.catalog.workload_store import WorkloadStore, init_db, normalize_retrieval_hits


def test_normalize_full_corpus_vs_sliced():
    corpus = [f"c{i}" for i in range(100)]
    sources = [f"s{i}" for i in range(100)]
    top_idxs = [2, 5, 9]
    scores = [0.9, 0.8, 0.7]
    page_map = {2: [10], 5: [11], 9: [12]}
    rows = normalize_retrieval_hits(top_idxs, corpus, sources, scores, page_map)
    assert len(rows) == 3
    assert rows[0]["chunk_idx"] == 2 and rows[0]["chunk_text_preview"] == "c2"
    assert rows[1]["page_number"] == 11

    log_chunks = [corpus[i] for i in top_idxs]
    log_sources = [sources[i] for i in top_idxs]
    rows2 = normalize_retrieval_hits(top_idxs, log_chunks, log_sources, scores, page_map)
    assert rows2 == rows


def test_record_chat_turn_transaction_and_fk(tmp_path: Path):
    db = tmp_path / "w.db"
    init_db(db)
    store = WorkloadStore(db)
    qid = store.record_chat_turn(
        query="What is a B-tree?",
        config_state={"top_k": 3},
        chat_request_params={"x": 1},
        top_idxs=[0, 1],
        chunks=["chunk zero", "chunk one", "chunk two"],
        sources=["a.md", "a.md", "b.md"],
        ordered_scores=[1.0, 0.5],
        page_map={0: [3], 1: [4]},
        full_response="A B-tree is ...",
        top_k=2,
        additional_log_info={"note": "test"},
    )
    assert qid == 1

    with sqlite3.connect(str(db)) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        nq = conn.execute("SELECT COUNT(*) FROM queries").fetchone()[0]
        nh = conn.execute("SELECT COUNT(*) FROM retrieval_hits").fetchone()[0]
        assert nq == 1 and nh == 2

        row = conn.execute(
            "SELECT query_text, config_json, additional_log_json FROM queries WHERE id = ?",
            (qid,),
        ).fetchone()
        assert row[0] == "What is a B-tree?"
        assert json.loads(row[1])["top_k"] == 3
        assert json.loads(row[2])["note"] == "test"

        h = conn.execute(
            "SELECT rank, chunk_idx, score, page_number FROM retrieval_hits WHERE query_id = ? ORDER BY rank",
            (qid,),
        ).fetchall()
        assert h[0] == (1, 0, 1.0, 3)
        assert h[1] == (2, 1, 0.5, 4)


def test_init_db_idempotent(tmp_path: Path):
    db = tmp_path / "x.db"
    init_db(db)
    init_db(db)
    init_db(db)
    assert db.exists()
