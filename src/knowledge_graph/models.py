from dataclasses import dataclass, field
from typing import Any
import yaml


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
class KGPipelineConfig:
    corpus_description: str = ""
    min_cooccurrence: int = 0
    top_n: int = 10

    @classmethod
    def from_yaml(cls, path: str) -> "KGPipelineConfig":
        """Load the ``kg_pipeline`` section from a project config YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        kg = dict(data.get("kg_pipeline", {}))
        return cls(**kg)
