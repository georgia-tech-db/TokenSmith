from dataclasses import dataclass, field
from typing import Any
from enum import Enum
from dataclasses import dataclass, field

import numpy as np
import yaml

from src.knowledge_graph.build import TOP_N


@dataclass
class Chunk:
    id: int
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractionResult:
    chunk_id: int
    keywords: list[str] = field(default_factory=list)


@dataclass
class CanonicalizationResult:
    synonym_table: dict[str, str]
    canonical_keywords: list[str]
    canonical_embeddings: np.ndarray
    stats: dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryFeatures:
    query_node_count: int = 0
    component_count: int = 0
    max_path_length: int = 0
    avg_path_length: float = 0.0
    avg_degree: float = 0.0
    max_degree: int = 0
    subgraph_node_count: int = 0
    subgraph_edge_count: int = 0
    doc_count: int = 0

    def to_dict(self) -> dict:
        return {
            "query_node_count": self.query_node_count,
            "component_count": self.component_count,
            "max_path_length": self.max_path_length,
            "avg_path_length": self.avg_path_length,
            "avg_degree": self.avg_degree,
            "max_degree": self.max_degree,
            "subgraph_node_count": self.subgraph_node_count,
            "subgraph_edge_count": self.subgraph_edge_count,
            "doc_count": self.doc_count,
        }


class DifficultyCategory(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class DifficultyComponents:
    multihop: int
    fragmentation: int
    subgraph_size: int
    branching: int
    dispersion: int

    def to_dict(self) -> dict:
        return {
            "multihop": self.multihop,
            "fragmentation": self.fragmentation,
            "subgraph_size": self.subgraph_size,
            "branching": self.branching,
            "dispersion": self.dispersion,
        }


@dataclass
class DifficultyScore:
    score: int
    category: DifficultyCategory
    components: DifficultyComponents

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "category": self.category.value,
            "components": self.components.__dict__,
        }


@dataclass
class QueryAnalysisResult:
    query: str
    features: QueryFeatures
    difficulty: DifficultyScore

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "features": self.features.to_dict(),
            "difficulty": self.difficulty.to_dict(),
        }


@dataclass
class RunMetadata:
    """Configuration and execution statistics for a pipeline run."""

    config: dict[str, Any] = field(default_factory=dict)
    statistics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "config": self.config,
            "statistics": self.statistics,
        }


@dataclass
class CanonicalizationConfig:
    llm_model: str = "openai/gpt-4o-mini"
    embed_model: str = "models/Qwen3-Embedding-4B-Q5_K_M.gguf"
    similarity_threshold: float = 0.78
    max_group_size: int = 30
    batch_size: int = 15


@dataclass
class KGPipelineConfig:
    corpus_description: str = ""
    min_cooccurrence: int = 0
    top_n: int = TOP_N
    canonicalization: CanonicalizationConfig = field(
        default_factory=CanonicalizationConfig
    )

    @classmethod
    def from_yaml(cls, path: str) -> "KGPipelineConfig":
        """Load the ``kg_pipeline`` section from a project config YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        kg = dict(data.get("kg_pipeline", {}))
        canon_data = kg.pop("canonicalization", {})
        return cls(**kg, canonicalization=CanonicalizationConfig(**canon_data))
