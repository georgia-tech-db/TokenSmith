"""Abstract base class for node/keyword extractors."""

from abc import ABC, abstractmethod
from typing import List

from src.knowledge_graph.models import Chunk, ExtractionResult


class BaseExtractor(ABC):
    """Extract and normalize nodes from each chunk."""

    @abstractmethod
    def extract(self, chunks: List[Chunk]) -> List[ExtractionResult]:
        """Extract node labels from *chunks*.

        Args:
            chunks: The chunks produced by a :class:`BaseDivider`.

        Returns:
            One :class:`ExtractionResult` per chunk, containing normalized
            node labels.
        """
        ...
