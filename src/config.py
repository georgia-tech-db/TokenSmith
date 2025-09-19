from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Callable, Any

import yaml

from src.chunking import ChunkStrategy, make_chunk_strategy, CharChunkConfig, TokenChunkConfig, SlidingTokenConfig, \
    SectionChunkConfig, ChunkConfig


@dataclass
class QueryPlanConfig:
    # chunking
    chunk_config: ChunkConfig

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
        return make_chunk_strategy(config=self.chunk_config)

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
    def from_yaml(path: os.PathLike) -> QueryPlanConfig:
        raw = yaml.safe_load(open(path))

        def pick(key, default=None):
            return raw.get(key, default)

        chunk_mode, chunk_config = QueryPlanConfig.get_chunk_config(raw)

        cfg = QueryPlanConfig(
            # Chunking
            chunk_config   = chunk_config,

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

    @staticmethod
    def get_chunk_config(raw):
        chunk_mode = raw.get("chunk_mode", "chars").lower()
        chunk_config = None
        if chunk_mode == "chars":
            chunk_config = CharChunkConfig(raw.get("chunk_size_char", 20_000))
        elif chunk_mode == "tokens":
            chunk_config = TokenChunkConfig(raw.get("chunk_tokens", 500))
        elif chunk_mode == "sliding-tokens":
            chunk_config = SlidingTokenConfig(
                max_tokens=raw.get("chunk_tokens", 350),
                overlap_tokens=raw.get("overlap_tokens", 80),
                tokenizer_name=raw.get("embed_model", "sentence-transformers/all-MiniLM-L6-v2"),
            )
        elif chunk_mode == "sections":
            chunk_config = SectionChunkConfig()
        else:
            raise ValueError(f"Unknown chunk_mode: {chunk_mode}")
        return chunk_mode, chunk_config

    def _validate(self) -> None:
        assert self.top_k > 0, "top_k must be > 0"
        assert self.pool_size >= self.top_k, "pool_size must be >= top_k"
        assert self.ensemble_method.lower() in {"linear","weighted","rrf"}
        if self.ensemble_method.lower() in {"linear","weighted"}:
            s = sum(self.ranker_weights.values()) or 1.0
            self.ranker_weights = {k: v/s for k, v in self.ranker_weights.items()}
        self.chunk_config.validate()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_config": self.chunk_config.to_string(),
            "index_prefix": self.index_prefix,
            "top_k": self.top_k,
            "pool_size": self.pool_size,
            "embed_model": self.embed_model,
            "ensemble_method": self.ensemble_method,
            "rrf_k": self.rrf_k,
            "ranker_weights": self.ranker_weights,
            "halo_mode": self.halo_mode,
            "max_gen_tokens": self.max_gen_tokens,
        }

