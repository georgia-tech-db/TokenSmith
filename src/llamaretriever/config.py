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

    # ── Embedding (same GGUF model as original pipeline) ───────────────
    embed_model: str = "models/Qwen3-Embedding-4B-Q5_K_M.gguf"
    embed_n_ctx: int = 4096

    # ── Generation (same GGUF model as original pipeline) ────────────────
    gen_model: str = "models/Qwen3-4B-Q5_K_M.gguf"
    gen_context_window: int = 8192
    max_gen_tokens: int = 1024
    gen_temperature: float = 0.2
    n_gpu_layers: int = -1  # -1 = offload all to GPU

    # ── Chunking (matches original: 2000 / 200) ─────────────────────────
    chunk_size: int = 2000
    chunk_overlap: int = 200

    # ── Retrieval (matches original: RRF fusion) ─────────────────────────
    num_candidates: int = 50  # per-retriever pool size
    top_k: int = 5            # final chunks after reranking

    # ── Reranking (same cross-encoder as original) ───────────────────────
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    use_reranker: bool = True

    # ── Agent ────────────────────────────────────────────────────────────
    max_curate_steps: int = 3  # max LLM calls for evidence curation

    # ── Factory ──────────────────────────────────────────────────────────
    @classmethod
    def from_yaml(cls, path: os.PathLike) -> "LlamaIndexConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid)

    def __post_init__(self) -> None:
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
