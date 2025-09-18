from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Callable
import yaml

from src.chunking import ChunkStrategy, make_chunk_strategy


@dataclass
class QueryPlanConfig:
    # chunking
    chunk_mode: str
    chunk_size_char: int
    chunk_tokens: int

    # retrieval + ranking
    index_prefix: str
    top_k: int
    pool_size: int
    embed_model: str

    ensemble_method: str
    rrf_k: int
    ranker_weights: Dict[str, float]
    halo_mode: str
    seg_filter: Callable

    # generation
    max_gen_tokens: int

    # ---------- chunking strategy + artifact name helpers ----------
    def make_strategy(self) -> ChunkStrategy:
        return make_chunk_strategy(
            self.chunk_mode,
            chunk_size_char=self.chunk_size_char,
            chunk_tokens=self.chunk_tokens,
            tokenizer_name=self.embed_model,
        )

    def get_faiss_prefix(self, out_prefix: str) -> str:
        strategy = self.make_strategy()
        os.makedirs(f"index/{strategy.artifact_folder_name()}", exist_ok=True)
        return f"index/{strategy.artifact_folder_name()}/{out_prefix}"

    def get_tfidf_prefix(self, out_prefix: str) -> str:
        strategy = self.make_strategy()
        os.makedirs(f"index/{strategy.artifact_folder_name()}/meta", exist_ok=True)
        return f"index/{strategy.artifact_folder_name()}/meta/{out_prefix}"

    # ---------- factory + validation ----------
    @staticmethod
    def from_yaml(path: str) -> QueryPlanConfig:
        raw = yaml.safe_load(open(path))

        def pick(key, default=None):
            return raw.get(key, default)

        cfg = QueryPlanConfig(
            # Chunking
            chunk_mode     = pick("chunk_mode", "chars"),
            chunk_size_char= pick("chunk_size_char", 20_000),
            chunk_tokens   = pick("chunk_tokens", 500),

            # Retrieval + Ranking
            index_prefix   = pick("index_prefix", "textbook_index"),
            top_k          = pick("top_k", 5),
            pool_size      = pick("pool_size", 60),
            embed_model    = pick("embed_model", "sentence-transformers/all-MiniLM-L6-v2"),
            ensemble_method= pick("ensemble_method", "rrf"),
            rrf_k          = pick("rrf_k", 60),
            ranker_weights = pick("ranker_weights", {"faiss":0.6,"bm25":0.4,"tf-idf":0}),
            max_gen_tokens = pick("max_gen_tokens", 400),
            halo_mode      = pick("halo_mode", "none"),
            seg_filter     = pick("seg_filter", None)
        )
        cfg._validate()
        return cfg

    def _validate(self) -> None:
        assert self.top_k > 0, "top_k must be > 0"
        assert self.pool_size >= self.top_k, "pool_size must be >= top_k"
        assert self.ensemble_method.lower() in {"linear","weighted","rrf"}
        if self.ensemble_method.lower() in {"linear","weighted"}:
            s = sum(self.ranker_weights.values()) or 1.0
            self.ranker_weights = {k: v/s for k, v in self.ranker_weights.items()}

        assert self.chunk_mode in {"chars", "tokens", "sliding-tokens", "sections"}, \
            f"Invalid chunk_mode: {self.chunk_mode}"

        # Chunking config sanity
        if self.chunk_mode == "chars":
            assert self.chunk_size_char > 0, "chunk_size_char must be > 0"
        if self.chunk_mode == "tokens":
            assert self.chunk_tokens > 0, "chunk_tokens must be > 0"
        if self.chunk_mode == "sliding-tokens":
            assert self.chunk_tokens > 0, "chunk_tokens must be > 0 for sliding-tokens"

