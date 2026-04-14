from abc import abstractmethod
from typing import Any

from src.knowledge_graph.models import Chunk, ExtractionResult


class BaseExtractor:
    """Extract keywords from each chunk."""

    def __init__(self):
        self.metadata: dict[str, Any] = {}

    def get_config(self) -> dict[str, Any]:
        """Return a dict describing this component's configuration.

        Subclasses should call ``super().get_config()`` and update the result
        with their own parameters.
        """
        return {"class": self.__class__.__name__}

    @abstractmethod
    def extract(self, chunks: list[Chunk]) -> list[ExtractionResult]:
        """Extract node labels from *chunks*.

        Args:
            chunks: The chunks produced by a :class:`BaseDivider`.

        Returns:
            A list of extraction results, each containing the chunk ID and a list of raw keywords.
        """
        ...
