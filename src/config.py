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
    
    # testing
    system_prompt_mode: str
    disable_chunks: bool
    use_golden_chunks: bool
    output_mode: str
    metrics: list

    # query enhancement
    use_hyde: bool
    hyde_max_tokens: int
    use_indexed_chunks: bool

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
        raw = yaml.safe_load(open(path))

        def pick(key, default=None):
            return raw.get(key, default)

        chunk_config = QueryPlanConfig.get_chunk_config(raw)

        cfg = QueryPlanConfig(
            # Chunking
            chunk_config   = chunk_config,

            # Retrieval + Ranking
            top_k          = pick("top_k", 5),
            pool_size      = pick("pool_size", 60),
            embed_model    = pick("embed_model", "sentence-transformers/all-MiniLM-L6-v2"),
            ensemble_method= pick("ensemble_method", "rrf"),
            rrf_k          = pick("rrf_k", 60),
            ranker_weights = pick("ranker_weights", {"faiss":0.6,"bm25":0.4}),
            max_gen_tokens = pick("max_gen_tokens", 400),
            rerank_mode    = pick("rerank_mode", "none"),
            seg_filter     = pick("seg_filter", None),
            model_path     = pick("model_path", None),
            
            # Testing
            system_prompt_mode = pick("system_prompt_mode", "baseline"),
            disable_chunks  = pick("disable_chunks", False),
            use_golden_chunks = pick("use_golden_chunks", False),
            output_mode    = pick("output_mode", "terminal"),
            metrics        = pick("metrics", ["all"]),
            use_indexed_chunks= pick("use_indexed_chunks", False),
            
            # Query Enhancement
            use_hyde       = pick("use_hyde", False),
            hyde_max_tokens= pick("hyde_max_tokens", 100),
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
            "model_path": self.model_path,
            "system_prompt_mode": self.system_prompt_mode,
            "disable_chunks": self.disable_chunks,
            "use_golden_chunks": self.use_golden_chunks,
            "output_mode": self.output_mode,
            "metrics": self.metrics,
            "use_indexed_chunks": self.use_indexed_chunks,
            "use_hyde": self.use_hyde,
            "hyde_max_tokens": self.hyde_max_tokens,
        }
