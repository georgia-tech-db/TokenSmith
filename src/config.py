from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass, field, fields
from typing import Dict, Iterable, Optional

import yaml

from src.preprocessing.chunking import ChunkConfig, ChunkStrategy, SectionRecursiveConfig, SectionRecursiveStrategy


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATHS = (
    pathlib.Path.home() / ".config" / "tokensmith" / "config.yaml",
    REPO_ROOT / "config" / "config.yaml",
)
LEGACY_CONFIG_ALIASES = {
    "pool_size": "num_candidates",
    "recursive_chunk_size": "chunk_size_in_chars",
    "chunk_size": "chunk_size_in_chars",
    "chunk_size_char": "chunk_size_in_chars",
    "recursive_overlap": "chunk_overlap",
    "system_prompt": "system_prompt_mode",
}


def resolve_config_path(requested: Optional[os.PathLike] = None) -> pathlib.Path:
    """Resolve the active TokenSmith config file using documented precedence."""
    if requested is not None:
        candidate = pathlib.Path(requested).expanduser()
        if not candidate.is_absolute():
            candidate = (pathlib.Path.cwd() / candidate).resolve()
        else:
            candidate = candidate.resolve()
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"TokenSmith config not found at {candidate}")

    for candidate in DEFAULT_CONFIG_PATHS:
        candidate = candidate.expanduser()
        if candidate.exists():
            return candidate.resolve()

    searched = "\n".join(f" - {path}" for path in DEFAULT_CONFIG_PATHS)
    raise FileNotFoundError(
        "TokenSmith config not found. Searched:\n"
        f"{searched}\n"
        "Pass --config to point at a config file explicitly."
    )


def _dedupe_paths(paths: Iterable[pathlib.Path]) -> list[pathlib.Path]:
    unique: list[pathlib.Path] = []
    seen: set[pathlib.Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            unique.append(resolved)
            seen.add(resolved)
    return unique


def _normalize_config_data(cls, data: Dict[str, object]) -> Dict[str, object]:
    normalized = dict(data)
    for legacy_key, canonical_key in LEGACY_CONFIG_ALIASES.items():
        if legacy_key not in normalized:
            continue
        legacy_value = normalized.pop(legacy_key)
        if canonical_key in normalized and normalized[canonical_key] != legacy_value:
            raise ValueError(
                f"Config specifies both '{legacy_key}' and '{canonical_key}' with different values."
            )
        normalized.setdefault(canonical_key, legacy_value)

    known_fields = {field.name for field in fields(cls) if field.init}
    unknown_keys = sorted(set(normalized) - known_fields)
    if unknown_keys:
        raise TypeError(
            "Unsupported config keys: "
            + ", ".join(unknown_keys)
        )
    return normalized

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

    _config_path: Optional[pathlib.Path] = field(default=None, init=False, repr=False, compare=False)

    # ---------- factory + validation ----------
    @classmethod
    def from_yaml(cls, path: os.PathLike) -> RAGConfig:
        """Create a RAGConfig from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            A fully validated RAGConfig instance.
        """
        config_path = pathlib.Path(path).expanduser().resolve()
        with config_path.open() as f:
            data = yaml.safe_load(f) or {}
        cfg = cls(**_normalize_config_data(cls, data))
        cfg._config_path = config_path
        cfg.resolve_paths()
        return cfg

    def __post_init__(self):
        """Validation logic runs automatically after initialization."""
        # Support README/CLI naming that uses model_path for the generation model.
        default_gen_model = next(
            field.default for field in fields(type(self)) if field.name == "gen_model"
        )
        if self.model_path and self.gen_model not in {None, self.model_path, default_gen_model}:
            raise ValueError("Config specifies both 'model_path' and 'gen_model' with different values.")
        if self.model_path:
            self.gen_model = self.model_path
        self.model_path = self.gen_model

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

    # ---------- path resolution + validation ----------

    @property
    def project_root(self) -> pathlib.Path:
        return REPO_ROOT

    @property
    def config_directory(self) -> pathlib.Path:
        if self._config_path is not None:
            return self._config_path.parent
        return REPO_ROOT

    def _search_roots(self) -> list[pathlib.Path]:
        return _dedupe_paths(
            [
                self.config_directory,
                REPO_ROOT,
                pathlib.Path.cwd(),
            ]
        )

    def _resolve_path(self, value: os.PathLike | str, *, model_file: bool = False) -> str:
        raw_path = pathlib.Path(os.fspath(value)).expanduser()
        if raw_path.is_absolute():
            return str(raw_path.resolve())

        candidates: list[pathlib.Path] = []
        for root in self._search_roots():
            candidates.append((root / raw_path).resolve())
            if model_file and len(raw_path.parts) == 1:
                candidates.append((root / "models" / raw_path.name).resolve())

        for candidate in candidates:
            if candidate.exists():
                return str(candidate)

        return str(candidates[0] if candidates else raw_path.resolve())

    def resolve_paths(self) -> RAGConfig:
        """Resolve config-relative and repo-relative paths to absolute paths."""
        self.embed_model = self._resolve_path(self.embed_model, model_file=True)
        self.gen_model = self._resolve_path(self.gen_model, model_file=True)
        self.model_path = self.gen_model
        self.extracted_index_path = self._resolve_path(self.extracted_index_path)
        self.page_to_chunk_map_path = self._resolve_path(self.page_to_chunk_map_path)
        return self

    def apply_overrides(
        self,
        *,
        model_path: Optional[str] = None,
        embed_model: Optional[str] = None,
    ) -> RAGConfig:
        """Apply CLI or runtime overrides and re-resolve any relative paths."""
        if model_path:
            self.gen_model = model_path
            self.model_path = model_path
        if embed_model:
            self.embed_model = embed_model
        return self.resolve_paths()

    def validate_runtime_files(
        self,
        *,
        require_embedding: bool = True,
        require_generation: bool = True,
        require_index_sidecars: bool = False,
    ) -> None:
        """Fail fast when the configured runtime assets are missing."""
        missing: list[str] = []

        checks: list[tuple[str, str]] = []
        if require_embedding:
            checks.append(("embedding model", self.embed_model))
        if require_generation:
            checks.append(("generation model", self.gen_model))
        if require_index_sidecars and self.ranker_weights.get("index_keywords", 0) > 0:
            checks.append(("extracted index", str(self.extracted_index_path)))
            checks.append(("page-to-chunk map", str(self.page_to_chunk_map_path)))

        for label, raw_path in checks:
            path = pathlib.Path(raw_path)
            if not path.exists():
                missing.append(f"{label}: {path}")
            elif not path.is_file():
                missing.append(f"{label}: {path} (not a file)")

        if missing:
            raise FileNotFoundError(
                "TokenSmith is missing required runtime assets:\n"
                + "\n".join(f" - {entry}" for entry in missing)
            )

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
        
        main_dir = REPO_ROOT / "index" / base_folder
        partial_dir = REPO_ROOT / "index" / f"partial_{base_folder}"

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
