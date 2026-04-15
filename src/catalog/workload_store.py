"""
SQLite workload store: one row per chat query plus retrieval hits.

Schema is created idempotently; foreign keys are enforced per connection.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union


def _page_for_chunk(page_map: Mapping[int, Any], chunk_idx: int) -> int:
    v = page_map.get(chunk_idx, 1)
    if isinstance(v, list) and v:
        return int(v[0])
    if isinstance(v, int):
        return v
    try:
        return int(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 1


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_dumps(obj: Any) -> Optional[str]:
    if obj is None:
        return None
    return json.dumps(obj, ensure_ascii=False, default=str)


def normalize_retrieval_hits(
    top_idxs: Sequence[int],
    chunks: Sequence[str],
    sources: Sequence[str],
    ordered_scores: Sequence[Union[float, int]],
    page_map: Mapping[int, Any],
) -> List[Dict[str, Any]]:
    """
    Build per-hit records aligned with rank order.

    Supports two call conventions:
    - Full corpus: len(chunks) >> len(top_idxs); use chunks[i] for each i in top_idxs.
    - Pre-sliced log lines: len(chunks) == len(top_idxs); chunk[j] belongs to top_idxs[j].
    """
    n = len(top_idxs)
    if n == 0:
        return []

    if len(chunks) == n and len(sources) == n:
        texts = list(chunks)
        srcs = list(sources)
        idxs = list(top_idxs)
    else:
        idxs = list(top_idxs)
        texts = [chunks[i] for i in idxs]
        srcs = [sources[i] for i in idxs]

    scores: List[Optional[float]] = []
    for j in range(n):
        if j < len(ordered_scores):
            scores.append(float(ordered_scores[j]))
        else:
            scores.append(None)

    rows: List[Dict[str, Any]] = []
    for rank, (idx, text, src, score) in enumerate(zip(idxs, texts, srcs, scores), start=1):
        page = _page_for_chunk(page_map, int(idx))
        preview = text[:500] if text else ""
        rows.append(
            {
                "rank": rank,
                "chunk_idx": int(idx),
                "score": score,
                "source_path": str(src) if src is not None else None,
                "page_number": page,
                "chunk_text_preview": preview,
            }
        )
    return rows


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            query_text TEXT NOT NULL,
            config_json TEXT,
            chat_request_params_json TEXT,
            top_k INTEGER NOT NULL,
            full_response TEXT,
            additional_log_json TEXT
        );

        CREATE TABLE IF NOT EXISTS retrieval_hits (
            query_id INTEGER NOT NULL,
            rank INTEGER NOT NULL,
            chunk_idx INTEGER NOT NULL,
            score REAL,
            source_path TEXT,
            page_number INTEGER,
            chunk_text_preview TEXT,
            PRIMARY KEY (query_id, rank),
            FOREIGN KEY (query_id) REFERENCES queries(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_queries_created_at ON queries(created_at);
        CREATE INDEX IF NOT EXISTS idx_retrieval_hits_chunk_idx ON retrieval_hits(chunk_idx);
        """
    )


def init_db(db_path: Path) -> None:
    """Create parent dirs and apply schema if missing."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db_path)) as conn:
        _ensure_schema(conn)
        conn.commit()


@dataclass
class WorkloadStore:
    """Append-only workload logging to SQLite."""

    db_path: Path

    def __post_init__(self) -> None:
        self.db_path = Path(self.db_path)

    def record_chat_turn(
        self,
        *,
        query: str,
        config_state: Dict[str, Any],
        chat_request_params: Optional[Dict[str, Any]],
        top_idxs: Sequence[int],
        chunks: Sequence[str],
        sources: Sequence[str],
        ordered_scores: Sequence[Union[float, int]],
        page_map: Mapping[int, Any],
        full_response: str,
        top_k: int,
        additional_log_info: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Insert one query row and its retrieval hits in a single transaction.
        Returns the new queries.id.
        """
        hits = normalize_retrieval_hits(
            top_idxs, chunks, sources, ordered_scores, page_map
        )
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(self.db_path)) as conn:
            _ensure_schema(conn)
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO queries (
                    created_at, query_text, config_json, chat_request_params_json,
                    top_k, full_response, additional_log_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _utc_iso(),
                    query,
                    _json_dumps(config_state),
                    _json_dumps(chat_request_params),
                    int(top_k),
                    full_response,
                    _json_dumps(additional_log_info),
                ),
            )
            qid = int(cur.lastrowid)
            for h in hits:
                cur.execute(
                    """
                    INSERT INTO retrieval_hits (
                        query_id, rank, chunk_idx, score, source_path,
                        page_number, chunk_text_preview
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        qid,
                        h["rank"],
                        h["chunk_idx"],
                        h["score"],
                        h["source_path"],
                        h["page_number"],
                        h["chunk_text_preview"],
                    ),
                )
            conn.commit()
            return qid


def record_chat_turn(db_path: Path, **kwargs: Any) -> int:
    """Convenience: one-shot write using a temporary WorkloadStore."""
    return WorkloadStore(db_path).record_chat_turn(**kwargs)
