"""Configuration for the BookRAG-style retrieval pipeline."""

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

    # ── Retrieval ────────────────────────────────────────────────────────
    num_candidates: int = 50
    top_k: int = 5

    # ── BookRAG section retrieval ────────────────────────────────────────
    section_top_k: int = 5
    section_summary_chars: int = 500
    max_leaves: int = 20

    # ── Retrieval grading / filtering ────────────────────────────────────
    use_retrieval_grader: bool = True
    retrieval_min_score: float = 0.18
    retrieval_min_keyword_hits: int = 1
    retrieval_keep_at_least: int = 8
    retrieval_max_after_grade: int = 20

    # ── Index-time LLM for KG extraction (never used at query time) ─────
    # "none" = heuristic only, "local" = larger local GGUF, "openrouter" = API
    index_llm_provider: str = "none"
    index_llm_model: str = ""
    index_llm_context_window: int = 16384
    index_llm_max_tokens: int = 2048
    index_llm_temperature: float = 0.1

    # ── Reranking ────────────────────────────────────────────────────────
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    use_reranker: bool = True

    @classmethod
    def from_yaml(cls, path: os.PathLike) -> "LlamaIndexConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid)

    def __post_init__(self) -> None:
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)

    @property
    def leaf_persist_dir(self) -> str:
        return str(Path(self.persist_dir) / "leaf")

    @property
    def section_persist_dir(self) -> str:
        return str(Path(self.persist_dir) / "section")

    @property
    def tree_path(self) -> str:
        return str(Path(self.persist_dir) / "tree.json")
