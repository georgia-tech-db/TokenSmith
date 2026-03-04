from abc import ABC, abstractmethod
from typing import Any

from src.knowledge_graph.models import Chunk, ExtractionResult


class BaseExtractor(ABC):
    """Extract and normalize nodes from each chunk."""

    def __init__(self):
        self.metadata: dict[str, Any] = {}

    def get_config(self) -> dict[str, Any]:
        """Return the configuration of this extractor."""
        return {
            "class": self.__class__.__name__,
        }

    @abstractmethod
    def extract(self, chunks: list[Chunk]) -> list[ExtractionResult]:
        """Extract node labels from *chunks*.

        Args:
            chunks: The chunks produced by a :class:`BaseDivider`.

        Returns:
            One :class:`ExtractionResult` per chunk, containing normalized
            node labels.
        """
        ...
