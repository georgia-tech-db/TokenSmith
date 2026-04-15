"""
Golden parity: SQLite workload rows match canonical retrieval for the same
inputs passed to RunLogger.save_chat_log (full-corpus vs pre-sliced API).

When JSON uses the short-form retrieved_chunks branch, those idx/score/chunk
lines must match retrieval_hits.
"""

import json
import sqlite3
from pathlib import Path

import pytest

from src.catalog.workload_store import normalize_retrieval_hits
from src.instrumentation.logging import RunLogger


@pytest.fixture
def isolated_logger(tmp_path, monkeypatch):
    """Avoid writing under repo ./logs during tests."""
    logger = RunLogger()
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(logger, "logs_dir", log_dir)
    return logger


def _load_latest_query_and_hits(db: Path):
    with sqlite3.connect(str(db)) as conn:
        qid = conn.execute("SELECT MAX(id) FROM queries").fetchone()[0]
        assert qid is not None
        qrow = conn.execute(
            "SELECT query_text, top_k, full_response FROM queries WHERE id = ?",
            (qid,),
        ).fetchone()
        hits = conn.execute(
            """
            SELECT rank, chunk_idx, score, source_path, page_number, chunk_text_preview
            FROM retrieval_hits WHERE query_id = ? ORDER BY rank
            """,
            (qid,),
        ).fetchall()
        return qid, qrow, hits


def test_parity_full_corpus_style_matches_normalize(isolated_logger, tmp_path):
    """CLI-style: len(chunks) >> len(top_idxs); DB matches normalize_retrieval_hits."""
    db = tmp_path / "workload.db"
    corpus = [f"chunk text {i}" for i in range(50)]
    sources = [f"src/{i % 3}.md" for i in range(50)]
    top_idxs = [7, 12, 3]
    scores = [0.42, 0.41, 0.40]
    page_map = {7: [100], 12: [101, 102], 3: [5]}

    expected = normalize_retrieval_hits(
        top_idxs, corpus, sources, scores, page_map
    )

    isolated_logger.save_chat_log(
        query="What is normalization?",
        chat_request_params=None,
        ordered_scores=scores,
        config_state={"top_k": 3},
        top_idxs=top_idxs,
        chunks=corpus,
        sources=sources,
        page_map=page_map,
        full_response="answer body",
        top_k=3,
        workload_db_path=db,
    )

    _, qrow, hits = _load_latest_query_and_hits(db)
    assert qrow[0] == "What is normalization?"
    assert qrow[1] == 3
    assert qrow[2] == "answer body"
    assert len(hits) == 3
    for exp, hit in zip(expected, hits):
        rank, chunk_idx, score, source_path, page_number, preview = hit
        assert rank == exp["rank"]
        assert chunk_idx == exp["chunk_idx"]
        if exp["score"] is not None:
            assert score == pytest.approx(exp["score"])
        else:
            assert score is None
        assert source_path == exp["source_path"]
        assert page_number == exp["page_number"]
        assert preview == exp["chunk_text_preview"]


def test_parity_api_sliced_matches_json_and_db(isolated_logger, tmp_path):
    """API-style: equal lengths; JSON retrieved_chunks and DB agree."""
    db = tmp_path / "workload.db"
    top_idxs = [40, 41]
    log_chunks = [f"text for {i}" for i in top_idxs]
    log_sources = ["paper.md", "paper.md"]
    scores = [1.1, 2.2]
    page_map = {40: [1], 41: [2]}

    isolated_logger.save_chat_log(
        query="Define ACID.",
        chat_request_params={"k": 1},
        ordered_scores=scores,
        config_state={"mode": "x"},
        top_idxs=top_idxs,
        chunks=log_chunks,
        sources=log_sources,
        page_map=page_map,
        full_response="ACID is ...",
        top_k=2,
        workload_db_path=db,
    )

    json_files = sorted((isolated_logger.logs_dir).glob("chat_*.json"))
    assert len(json_files) == 1
    with open(json_files[0], encoding="utf-8") as f:
        payload = json.load(f)
    assert "retrieved_chunks" in payload
    rc = payload["retrieved_chunks"]
    assert len(rc) == 2

    _, _, hits = _load_latest_query_and_hits(db)
    assert len(hits) == 2
    for row, j, exp_rank in zip(hits, rc, (1, 2)):
        rank, chunk_idx, score, source_path, page_number, preview = row
        assert rank == exp_rank == j["rank"]
        assert chunk_idx == j["idx"]
        assert score == pytest.approx(j["score"])
        assert source_path == j["source"]
        # JSON may store list pages (get_page_numbers shape); DB stores first page int.
        j_page = j["page_number"]
        if isinstance(j_page, list):
            j_page = int(j_page[0]) if j_page else 1
        assert page_number == j_page
        assert preview == j["chunk"][:500]


def test_parity_scores_shorter_than_topk_zero_fills_scores(isolated_logger, tmp_path):
    """If ordered_scores is shorter than top_idxs, remaining scores are None in DB."""
    db = tmp_path / "workload.db"
    corpus = ["a", "b", "c"]
    sources = ["x.md", "x.md", "x.md"]
    top_idxs = [0, 2]
    scores = [0.9]  # only one score
    page_map = {0: [1], 2: [2]}

    expected = normalize_retrieval_hits(
        top_idxs, corpus, sources, scores, page_map
    )
    assert expected[1]["score"] is None

    isolated_logger.save_chat_log(
        query="q",
        chat_request_params=None,
        ordered_scores=scores,
        config_state={},
        top_idxs=top_idxs,
        chunks=corpus,
        sources=sources,
        page_map=page_map,
        full_response="r",
        top_k=2,
        workload_db_path=db,
    )
    _, _, hits = _load_latest_query_and_hits(db)
    assert hits[0][2] == pytest.approx(0.9)
    assert hits[1][2] is None
