"""Abstract base class for graph persisters."""

from abc import ABC, abstractmethod
from typing import List

import networkx as nx

from src.knowledge_graph.models import Chunk


class BasePersister(ABC):
    """Save the graph and chunk store to disk."""

    @abstractmethod
    def persist(self, graph: nx.Graph, chunks: List[Chunk], output_dir: str) -> None:
        """Persist *graph* and *chunks* to *output_dir*.

        Args:
            graph: The knowledge graph built by a :class:`BaseLinker`.
            chunks: The original chunks (for building the chunk store).
            output_dir: Directory to write output files into.
        """
        ...
