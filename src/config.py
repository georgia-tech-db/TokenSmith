from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Any

import yaml
import pathlib

from src.preprocessing.chunking import ChunkStrategy, SectionRecursiveStrategy, SectionRecursiveConfig, ChunkConfig

@dataclass
class RAGConfig:
    # chunking
    chunk_config: ChunkConfig = field(init=False)
    chunk_mode: str = "recursive_sections"
    chunk_size: int = 2000
    chunk_overlap: int = 200

    # retrieval + ranking
    top_k: int = 10
    num_candidates: int = 60
    embed_model: str = "models/Qwen3-Embedding-4B-Q5_K_M.gguf"
    ensemble_method: str = "rrf"
    rrf_k: int  = 60
    ranker_weights: Dict[str, float] = field(
        default_factory=lambda: {"faiss": 1.0, "bm25": 0.0, "index_keywords": 0.0}
    )
    rerank_mode: str = ""
    rerank_top_k: int = 5

    # generation
    max_gen_tokens: int = 400
    gen_model: str = "models/qwen2.5-3b-instruct-q8_0.gguf"
    
    # testing
    system_prompt_mode: str = "baseline"
    disable_chunks: bool = False
    use_golden_chunks: bool = False
    output_mode: str = "terminal"
    metrics: list = field(default_factory=lambda: ["all"])

    # query enhancement
    use_hyde: bool = False
    hyde_max_tokens: int = 300
    
    # index parameters
    use_indexed_chunks: bool = False
    extracted_index_path: os.PathLike = "data/extracted_index.json"
    page_to_chunk_map_path: os.PathLike = "index/sections/textbook_index_page_to_chunk_map.json"

    # ---------- factory + validation ----------
    @classmethod
    def from_yaml(cls, path: os.PathLike) -> RAGConfig:
        with open(path, 'r') as f:
            data = yaml.safe_load(open(path))
        return cls(**data)
    
    def __post_init__(self):
        """Validation logic runs automatically after initialization."""
        assert self.top_k > 0, "top_k must be > 0"
        assert self.num_candidates >= self.top_k, "num_candidates must be >= top_k"
        assert self.ensemble_method.lower() in {"linear","weighted","rrf"}
        if self.ensemble_method.lower() in {"linear","weighted"}:
            s = sum(self.ranker_weights.values()) or 1.0
            self.ranker_weights = {k: v/s for k, v in self.ranker_weights.items()}
        self.chunk_config = self.get_chunk_config()
        self.chunk_config.validate()

    # ---------- chunking + artifact name helpers ----------

    def get_chunk_config(self) -> ChunkConfig:
        """Parse chunk configuration from YAML."""
        if self.chunk_mode == "recursive_sections":
            return SectionRecursiveConfig(
                recursive_chunk_size=self.chunk_size,
                recursive_overlap=self.chunk_overlap
            )
        else:
            raise ValueError(f"Unknown chunk_mode: {self.chunk_mode}. Supported: recursive_sections")

    def get_chunk_strategy(self) -> ChunkStrategy:
        if isinstance(self.chunk_config, SectionRecursiveConfig):
            return SectionRecursiveStrategy(self.chunk_config)
        raise ValueError(f"Unknown chunk config type: {self.chunk_config.__class__.__name__}")

    def get_artifacts_directory(self) -> os.PathLike:
        """Returns the path prefix for index artifacts."""
        strategy = self.get_chunk_strategy()
        strategy_dir = pathlib.Path("index", strategy.artifact_folder_name())
        strategy_dir.mkdir(parents=True, exist_ok=True)
        return strategy_dir
    
    def get_config_state(self) -> None:
        """Returns dict of all config parameters except chunk_config """
        state = self.__dict__.copy()
        state.pop("chunk_config", None) # remove chunk_config to avoid serialization issues
        # also pop any non-serializable fields if needed
        for key in list(state.keys()):
            if not isinstance(state[key], (int, float, str, bool, list, dict, type(None))):
                state.pop(key)
        return state
        
