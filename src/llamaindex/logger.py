"""
JSON query logger.

Writes one pretty-printed .json file per session to logs/llamaindex/<timestamp>.json
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import LlamaIndexConfig


class QueryLogger:
    """Pretty-printed JSON logger for query diagnostics."""

    def __init__(self, cfg: LlamaIndexConfig) -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._path = Path(cfg.log_dir) / f"run_{ts}.json"
        self._data = {
            "session_start": ts,
            "config": {
                "embed_model": cfg.embed_model,
                "gen_model": cfg.gen_model,
                "chunk_size": cfg.chunk_size,
                "chunk_overlap": cfg.chunk_overlap,
                "num_candidates": cfg.num_candidates,
                "top_k": cfg.top_k,
                "rerank_model": cfg.rerank_model if cfg.use_reranker else None,
            },
            "queries": [],
        }
        self._flush()

    def log_query(
        self,
        question: str,
        answer: str,
        chunks: list[dict[str, Any]],
        retrieval_time_s: float,
        generation_time_s: float,
    ) -> None:
        """Log a single query with its chunks and timings."""
        self._data["queries"].append({
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer,
            "num_chunks": len(chunks),
            "chunks": chunks,
            "retrieval_time_s": round(retrieval_time_s, 3),
            "generation_time_s": round(generation_time_s, 3),
            "total_time_s": round(retrieval_time_s + generation_time_s, 3),
        })
        self._flush()

    def _flush(self) -> None:
        with open(self._path, "w") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)
