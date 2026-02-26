from abc import ABC, abstractmethod

import networkx as nx

from src.knowledge_graph.models import ExtractionResult


class BaseLinker(ABC):
    """Build a graph by creating edges between co-occurring nodes."""

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
