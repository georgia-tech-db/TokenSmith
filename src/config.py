from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Callable, Any

import yaml
import pathlib

from src.preprocessing.chunking import ChunkStrategy, make_chunk_strategy, SectionRecursiveConfig, ChunkConfig


@dataclass
class QueryPlanConfig:
    # chunking
    chunk_config: ChunkConfig

    # retrieval + ranking
    top_k: int
    pool_size: int
    embed_model: str

    ensemble_method: str
    rrf_k: int
    ranker_weights: Dict[str, float]
    rerank_mode: str
    seg_filter: Callable

    # generation
    max_gen_tokens: int
    
    model_path: os.PathLike

    # ---------- chunking strategy + artifact name helpers ----------
    def make_strategy(self) -> ChunkStrategy:
        return make_chunk_strategy(config=self.chunk_config)

    def make_artifacts_directory(self) -> os.PathLike:
        """Returns the path prefix for index artifacts."""
        strategy = self.make_strategy()
        strategy_dir = pathlib.Path("index", strategy.artifact_folder_name())
        strategy_dir.mkdir(parents=True, exist_ok=True)
        return strategy_dir

    # ---------- factory + validation ----------
    @staticmethod
    def from_yaml(path: os.PathLike) -> QueryPlanConfig:
        raw_config = yaml.safe_load(open(path))
        chunk_config = QueryPlanConfig.get_chunk_config(raw_config)

        cfg = QueryPlanConfig(
            # Chunking
            chunk_config   = chunk_config,

            # Retrieval + Ranking
            top_k          = raw_config.get("top_k", 5),
            pool_size      = raw_config.get("pool_size", 60),
            embed_model    = raw_config.get("embed_model", "sentence-transformers/all-MiniLM-L6-v2"),
            ensemble_method= raw_config.get("ensemble_method", "rrf"),
            rrf_k          = raw_config.get("rrf_k", 60),
            ranker_weights = raw_config.get("ranker_weights", {"faiss":0.6,"bm25":0.4}),
            max_gen_tokens = raw_config.get("max_gen_tokens", 400),
            rerank_mode    = raw_config.get("rerank_mode", "none"),
            seg_filter     = raw_config.get("seg_filter", None),
            model_path     = raw_config.get("model_path", None)
        )
        cfg._validate()
        return cfg

    @staticmethod
    def get_chunk_config(raw: Any) -> ChunkConfig:
        """Parse chunk configuration from YAML."""
        chunk_mode = raw.get("chunk_mode", "sections").lower()
        
        if chunk_mode == "sections":
            return SectionRecursiveConfig(
                recursive_chunk_size=raw.get("recursive_chunk_size", 1000),
                recursive_overlap=raw.get("recursive_overlap", 0)
            )
        else:
            raise ValueError(f"Unknown chunk_mode: {chunk_mode}. Only 'sections' is supported.")

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
            "top_k": self.top_k,
            "pool_size": self.pool_size,
            "embed_model": self.embed_model,
            "ensemble_method": self.ensemble_method,
            "rrf_k": self.rrf_k,
            "ranker_weights": self.ranker_weights,
            "rerank_mode": self.rerank_mode,
            "max_gen_tokens": self.max_gen_tokens,
            "model_path": self.model_path
        }
