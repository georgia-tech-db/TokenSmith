"""Configuration for the LlamaRetriever pipeline.

Uses the same retrieval stack (vector + BM25 + RRF + reranker) with an
iterative evidence-curation agent instead of single-shot generation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class LlamaIndexConfig:

    # ── Paths ────────────────────────────────────────────────────────────
    data_dir: str = "data"
    persist_dir: str = "index/llamaretriever"
    log_dir: str = "logs/llamaretriever"

    # ── Embedding ────────────────────────────────────────────────────────
    embed_model: str = "models/Qwen3-Embedding-4B-Q5_K_M.gguf"
    embed_n_ctx: int = 4096

    # ── Generation ───────────────────────────────────────────────────────
    gen_model: str = "models/Qwen3-4B-Q5_K_M.gguf"
    gen_context_window: int = 8192
    max_gen_tokens: int = 1024
    gen_temperature: float = 0.2
    n_gpu_layers: int = -1

    # ── Chunking ─────────────────────────────────────────────────────────
    chunk_size: int = 2000
    chunk_overlap: int = 200

    # ── Contextual chunk enrichment ──────────────────────────────────────
    enrich_chunks_with_context: bool = True
    max_context_prefix_chars: int = 350

    # ── Retrieval ────────────────────────────────────────────────────────
    num_candidates: int = 50
    top_k: int = 5

    # ── Retrieval grading / filtering ────────────────────────────────────
    use_retrieval_grader: bool = True
    retrieval_min_score: float = 0.18
    retrieval_min_keyword_hits: int = 1
    retrieval_keep_at_least: int = 8
    retrieval_max_after_grade: int = 20

    # ── Reranking ────────────────────────────────────────────────────────
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    use_reranker: bool = True

    # ── Agent ────────────────────────────────────────────────────────────
    max_curate_steps: int = 3

    @classmethod
    def from_yaml(cls, path: os.PathLike) -> "LlamaIndexConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid)

    def __post_init__(self) -> None:
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
