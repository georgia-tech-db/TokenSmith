from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Optional

import yaml
import pathlib

from src.preprocessing.chunking import ChunkStrategy, SectionRecursiveStrategy, SectionRecursiveConfig, ChunkConfig

@dataclass
class RAGConfig:
    """Central configuration dataclass for the RAG pipeline.

    Holds all tunable parameters for chunking, retrieval, ranking, generation,
    query enhancement, conversational memory, and adaptive retrieval.  Instances
    are typically created from a YAML file via ``from_yaml``.
    """

    # chunking
    chunk_config: ChunkConfig = field(init=False)
    chunk_mode: str = "recursive_sections"
    chunk_size_in_chars: int = 2000
    chunk_overlap: int = 300

    # retrieval + ranking
    top_k: int = 10
    num_candidates: int = 60
    embed_model: str = "models/embedders/Qwen3-Embedding-4B-Q5_K_M.gguf"
    embedding_model_context_window: int = 4096
    ensemble_method: str = "rrf"
    rrf_k: int = 60
    ranker_weights: Dict[str, float] = field(
        default_factory=lambda: {"faiss": 1.0, "bm25": 0.0, "index_keywords": 0.0}
    )
    rerank_mode: str = ""
    rerank_top_k: int = 5

    # generation
    max_gen_tokens: int = 400
    gen_model: str = "models/generators/qwen2.5-3b-instruct-q8_0.gguf"
    model_path: Optional[str] = None
    # testing
    system_prompt_mode: str = "baseline"
    disable_chunks: bool = False
    use_golden_chunks: bool = False
    output_mode: str = "terminal"
    metrics: list = field(default_factory=lambda: ["all"])

    # query enhancement
    use_hyde: bool = False
    hyde_max_tokens: int = 300
    use_double_prompt: bool = False

    # cache
    semantic_cache_enabled: bool = False
    semantic_cache_bi_encoder_threshold: float = 0.90
    semantic_cache_cross_encoder_threshold: float = 0.99

    # conversational memory
    enable_history: bool = True
    max_history_turns: int = 3

    # index parameters
    use_indexed_chunks: bool = False
    extracted_index_path: os.PathLike = "data/extracted_index.json"
    page_to_chunk_map_path: os.PathLike = "index/sections/textbook_index_page_to_chunk_map.json"

    # adaptive retrieval
    section_top_k: int = 4
    page_rerank_window: int = 20
    decomposition_max_subqueries: int = 4
    enable_adaptive_routing: bool = True
    enable_hierarchical_retrieval: bool = True
    retrieval_confidence_threshold: float = 0.03
    fallback_candidate_multiplier: int = 2

    # user feedback modeling
    enable_topic_extraction: bool = False

    # ---------- factory + validation ----------
    @classmethod
    def from_yaml(cls, path: os.PathLike) -> RAGConfig:
        """Create a RAGConfig from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            A fully validated RAGConfig instance.
        """
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def __post_init__(self):
        """Validation logic runs automatically after initialization."""
        # Support README/CLI naming that uses model_path for the generation model.
        if self.model_path:
            self.gen_model = self.model_path

        assert self.top_k > 0, "top_k must be > 0"
        assert self.num_candidates >= self.top_k, "num_candidates must be >= top_k"
        assert self.section_top_k > 0, "section_top_k must be > 0"
        assert self.page_rerank_window >= self.top_k, "page_rerank_window must be >= top_k"
        assert self.decomposition_max_subqueries > 0, "decomposition_max_subqueries must be > 0"
        assert self.retrieval_confidence_threshold >= 0, "retrieval_confidence_threshold must be >= 0"
        assert self.fallback_candidate_multiplier >= 1, "fallback_candidate_multiplier must be >= 1"
        assert self.ensemble_method.lower() in {"linear","weighted","rrf"}
        assert self.embedding_model_context_window > 0, "embedding_model_context_window must be > 0"
        if self.ensemble_method.lower() in {"linear","weighted"}:
            s = sum(self.ranker_weights.values()) or 1.0
            self.ranker_weights = {k: v / s for k, v in self.ranker_weights.items()}
        self.chunk_config = self.get_chunk_config()
        self.chunk_config.validate()

    # ---------- chunking + artifact name helpers ----------

    def get_chunk_config(self) -> ChunkConfig:
        """Parse chunk configuration from YAML."""
        if self.chunk_mode == "recursive_sections":
            return SectionRecursiveConfig(
                recursive_chunk_size=self.chunk_size_in_chars,
                recursive_overlap=self.chunk_overlap,
            )
        else:
            raise ValueError(f"Unknown chunk_mode: {self.chunk_mode}. Supported: recursive_sections")

    def get_chunk_strategy(self) -> ChunkStrategy:
        """Return the ChunkStrategy corresponding to the current chunk config."""
        if isinstance(self.chunk_config, SectionRecursiveConfig):
            return SectionRecursiveStrategy(self.chunk_config)
        raise ValueError(f"Unknown chunk config type: {self.chunk_config.__class__.__name__}")

    def get_artifacts_directory(self, partial: bool = False) -> os.PathLike:
        """
        Returns the path prefix for index artifacts.
        If partial=True, strictly returns the partial directory.
        If partial=False, returns the main directory if it exists, 
        otherwise falls back to the partial directory.
        """
        strategy = self.get_chunk_strategy()
        base_folder = strategy.artifact_folder_name()
        
        main_dir = pathlib.Path("index", base_folder)
        partial_dir = pathlib.Path("index", f"partial_{base_folder}")

        if partial:
            target_dir = partial_dir
            print("Using partial directory (change partial to false in config.yaml to use full directory)")
        else:
            # Fallback logic: use main if it exists, otherwise use partial if it exists
            if main_dir.exists():
                target_dir = main_dir
            elif partial_dir.exists():
                target_dir = partial_dir
                print("Using partial directory (unable to find full directory)")
            else:
                target_dir = main_dir

        target_dir.mkdir(parents=True, exist_ok=True)
        return target_dir

    def get_page_to_chunk_map_path(self, artifacts_dir: os.PathLike, index_prefix: str) -> os.PathLike:
        """Returns the path to the page-to-chunk map file."""
        return pathlib.Path(artifacts_dir) / f"{index_prefix}_page_to_chunk_map.json"
    
    def get_config_state(self) -> Dict[str, object]:
        """Return a serializable dict of all config parameters except chunk_config."""
        state = self.__dict__.copy()
        state.pop("chunk_config", None)
        for key in list(state.keys()):
            if not isinstance(state[key], (int, float, str, bool, list, dict, type(None))):
                state.pop(key)
        return state

    def to_dict(self) -> Dict[str, object]:
        """Alias for ``get_config_state()``; returns the config as a plain dict."""
        return self.get_config_state()
