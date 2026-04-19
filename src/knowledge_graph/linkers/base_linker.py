from abc import abstractmethod
from typing import Any
import networkx as nx

from src.knowledge_graph.models import ExtractionResult


class BaseLinker:
    """Build a graph by creating edges between co-occurring nodes."""

    def __init__(self):
        self.metadata: dict[str, Any] = {}

    def get_config(self) -> dict[str, Any]:
        """Return a dict describing this component's configuration.

        Subclasses should call ``super().get_config()`` and update the result
        with their own parameters.
        """
        return {"class": self.__class__.__name__}

    @abstractmethod
    def link(self, extractions: list[ExtractionResult]) -> nx.Graph:
        """Create a :class:`networkx.Graph` from extraction results.

        Args:
            extractions: One :class:`ExtractionResult` per chunk.

        Returns:
            A NetworkX graph whose nodes carry ``chunk_ids`` and whose edges
            carry ``weight`` and ``chunk_ids`` attributes.
        """
        ...
