from abc import abstractmethod

from src.knowledge_graph.base import BasePipelineComponent
from src.knowledge_graph.models import Chunk, ExtractionResult


class BaseExtractor(BasePipelineComponent):
    """Extract keywords from each chunk."""

    @abstractmethod
    def extract(self, chunks: list[Chunk]) -> list[ExtractionResult]:
        """Extract node labels from *chunks*.

        Args:
            chunks: The chunks produced by a :class:`BaseDivider`.

        Returns:
            A list of extraction results, each containing the chunk ID and a list of raw keywords.
        """
        ...
