"""
Configuration for the LlamaIndex RAG pipeline.

Mirrors the structure of the original TokenSmith RAGConfig but adapted
for LlamaIndex components.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class LlamaIndexConfig:
    """All knobs for the LlamaIndex pipeline in one place."""

    # ── Paths ────────────────────────────────────────────────────────────
    data_dir: str = "data"
    persist_dir: str = "index/llamaindex"

    # ── Embedding model (HuggingFace, <5B params) ───────────────────────
    embed_model_name: str = "BAAI/bge-base-en-v1.5"  # 110M params
    embed_batch_size: int = 64
    embed_device: str = "cuda"  # "cpu", "cuda", "mps"

    # ── Generation model (local GGUF via llama-cpp) ─────────────────────
    gen_model_path: str = "models/qwen2.5-1.5b-instruct-q5_k_m.gguf"
    gen_context_window: int = 4096
    gen_max_tokens: int = 400
    gen_temperature: float = 0.2
    n_gpu_layers: int = -1  # -1 = offload all to GPU, 0 = CPU only

    # ── Chunking ─────────────────────────────────────────────────────────
    chunk_size: int = 1024
    chunk_overlap: int = 128

    # ── Retrieval ────────────────────────────────────────────────────────
    similarity_top_k: int = 10        # candidates from vector search
    keyword_top_k: int = 10           # candidates from keyword search
    final_top_k: int = 5              # after reranking

    # ── Reranking ────────────────────────────────────────────────────────
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-2-v2"
    use_reranker: bool = True

    # ── System prompt ────────────────────────────────────────────────────
    system_prompt: str = (
        "You are a helpful assistant. Answer the question using the provided "
        "context excerpts. If the context doesn't contain the answer, say so. "
        "Be concise and accurate."
    )

    # ── Factory ──────────────────────────────────────────────────────────
    @classmethod
    def from_yaml(cls, path: os.PathLike) -> "LlamaIndexConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        # Only keep keys that match our fields
        valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid)

    def __post_init__(self) -> None:
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
