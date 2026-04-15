from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def normalize_query(query: str) -> str:
    """Normalize query text so equivalent strings map to the same cache key."""
    return " ".join((query or "").strip().lower().split())


def _json_dumps_sorted(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def build_retrieval_signature(cfg) -> Dict[str, Any]:
    """
    Build a stable signature for retrieval-related settings.
    This intentionally excludes generation-only settings.
    """
    return {
        "chunk_mode": cfg.chunk_mode,
        "top_k": int(cfg.top_k),
        "num_candidates": int(cfg.num_candidates),
        "ensemble_method": cfg.ensemble_method,
        "ranker_weights": dict(sorted((cfg.ranker_weights or {}).items())),
        "rrf_k": int(cfg.rrf_k),
        "rerank_mode": cfg.rerank_mode,
        "rerank_top_k": int(cfg.rerank_top_k),
        "use_hyde": bool(cfg.use_hyde),
        "hyde_max_tokens": int(cfg.hyde_max_tokens),
        "use_indexed_chunks": bool(cfg.use_indexed_chunks),
        "disable_chunks": bool(cfg.disable_chunks),
        "embed_model": cfg.embed_model,
        "extracted_index_path": str(cfg.extracted_index_path),
        "page_to_chunk_map_path": str(cfg.page_to_chunk_map_path),
    }


def build_index_fingerprint(chunks: List[str], sources: Optional[List[str]], metadata: Optional[List[Dict[str, Any]]]) -> str:
    """
    Compute a lightweight corpus fingerprint from loaded artifacts.
    We hash corpus size plus stable previews rather than full content for speed.
    """
    hasher = hashlib.sha256()

    chunk_count = len(chunks or [])
    source_count = len(sources or [])
    meta_count = len(metadata or [])
    hasher.update(f"{chunk_count}|{source_count}|{meta_count}".encode("utf-8"))

    if chunks:
        # First/last/mid sampling keeps this lightweight while tracking major index changes.
        sample_idxs = {0, chunk_count // 2, max(chunk_count - 1, 0)}
        for idx in sorted(i for i in sample_idxs if 0 <= i < chunk_count):
            txt = chunks[idx] or ""
            hasher.update(f"{idx}:{len(txt)}:{txt[:256]}".encode("utf-8"))

    if metadata:
        for i in (0, len(metadata) // 2, len(metadata) - 1):
            if 0 <= i < len(metadata):
                page_numbers = metadata[i].get("page_numbers")
                hasher.update(f"m{i}:{page_numbers}".encode("utf-8"))

    return hasher.hexdigest()


def build_cache_key(normalized_query: str, retrieval_signature: Dict[str, Any], index_fingerprint: str) -> str:
    raw = f"{normalized_query}|{_json_dumps_sorted(retrieval_signature)}|{index_fingerprint}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass
class RetrievalCacheRecord:
    topk_idxs: List[int]
    ranked_chunks: List[str]
    ordered_scores: List[float]


class RetrievalContextCache:
    """
    Persistent cache for final retrieved context used for generation.
    Backed by SQLite for process-safe reuse.
    """

    def __init__(self, db_path: str = "index/cache/retrieval_cache.db", max_entries: int = 5000):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_entries = int(max_entries)
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS retrieval_context_cache (
                    cache_key TEXT PRIMARY KEY,
                    normalized_query TEXT NOT NULL,
                    retrieval_signature TEXT NOT NULL,
                    index_fingerprint TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_accessed TEXT NOT NULL,
                    hit_count INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_rcc_last_accessed ON retrieval_context_cache(last_accessed)"
            )

    def get(self, cache_key: str) -> Optional[RetrievalCacheRecord]:
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute(
                    """
                    SELECT payload_json
                    FROM retrieval_context_cache
                    WHERE cache_key=?
                    """,
                    (cache_key,),
                ).fetchone()
                if not row:
                    return None

                payload = json.loads(row[0])
                now = datetime.now(timezone.utc).isoformat()
                conn.execute(
                    """
                    UPDATE retrieval_context_cache
                    SET last_accessed=?, hit_count=hit_count+1
                    WHERE cache_key=?
                    """,
                    (now, cache_key),
                )
                conn.commit()

                return RetrievalCacheRecord(
                    topk_idxs=[int(i) for i in payload.get("topk_idxs", [])],
                    ranked_chunks=[str(c) for c in payload.get("ranked_chunks", [])],
                    ordered_scores=[float(s) for s in payload.get("ordered_scores", [])],
                )

    def set(
        self,
        cache_key: str,
        normalized_query: str,
        retrieval_signature: Dict[str, Any],
        index_fingerprint: str,
        record: RetrievalCacheRecord,
    ) -> None:
        payload = {
            "topk_idxs": [int(i) for i in record.topk_idxs],
            "ranked_chunks": [str(c) for c in record.ranked_chunks],
            "ordered_scores": [float(s) for s in record.ordered_scores],
        }
        now = datetime.now(timezone.utc).isoformat()

        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO retrieval_context_cache (
                        cache_key,
                        normalized_query,
                        retrieval_signature,
                        index_fingerprint,
                        payload_json,
                        created_at,
                        last_accessed,
                        hit_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, COALESCE(
                        (SELECT hit_count FROM retrieval_context_cache WHERE cache_key=?), 0
                    ))
                    """,
                    (
                        cache_key,
                        normalized_query,
                        _json_dumps_sorted(retrieval_signature),
                        index_fingerprint,
                        json.dumps(payload, ensure_ascii=False),
                        now,
                        now,
                        cache_key,
                    ),
                )
                conn.commit()

        self._prune_if_needed()

    def _prune_if_needed(self) -> None:
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute("SELECT COUNT(*) FROM retrieval_context_cache").fetchone()
                current_size = int(row[0]) if row else 0
                if current_size <= self.max_entries:
                    return
                to_delete = current_size - self.max_entries
                conn.execute(
                    """
                    DELETE FROM retrieval_context_cache
                    WHERE cache_key IN (
                        SELECT cache_key
                        FROM retrieval_context_cache
                        ORDER BY last_accessed ASC
                        LIMIT ?
                    )
                    """,
                    (to_delete,),
                )
                conn.commit()
