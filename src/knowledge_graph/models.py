from dataclasses import dataclass, field
from typing import Any
from enum import Enum


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
