"""
JSON query logger for the evidence-curation agent.

Writes one pretty-printed .json file per session to logs/llamaretriever/<timestamp>.json
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import LlamaIndexConfig


class QueryLogger:
    """Pretty-printed JSON logger for agent diagnostics."""

    def __init__(self, cfg: LlamaIndexConfig) -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._path = Path(cfg.log_dir) / f"run_{ts}.json"
        self._data: dict[str, Any] = {
            "session_start": ts,
            "config": {
                "embed_model": cfg.embed_model,
                "gen_model": cfg.gen_model,
                "chunk_size": cfg.chunk_size,
                "num_candidates": cfg.num_candidates,
                "top_k": cfg.top_k,
                "max_curate_steps": cfg.max_curate_steps,
                "rerank_model": cfg.rerank_model if cfg.use_reranker else None,
            },
            "queries": [],
        }
        self._flush()

    def log_query(
        self,
        question: str,
        answer: str,
        references: list[dict[str, Any]],
        iterations: list[dict[str, Any]],
        total_llm_calls: int,
        total_time_s: float,
        keywords: list[str] | None = None,
    ) -> None:
        """Log a single agent query with references, iterations, and timings."""
        entry: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer,
            "num_references": len(references),
            "references": references,
            "total_llm_calls": total_llm_calls,
            "total_time_s": round(total_time_s, 3),
            "iterations": iterations,
        }
        if keywords is not None:
            entry["keywords"] = keywords
        self._data["queries"].append(entry)
        self._flush()

    def _flush(self) -> None:
        with open(self._path, "w") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)
