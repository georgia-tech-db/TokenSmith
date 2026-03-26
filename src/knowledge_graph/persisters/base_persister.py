from abc import abstractmethod

import networkx as nx

from src.knowledge_graph.base import BasePipelineComponent
from src.knowledge_graph.models import Chunk, RunMetadata


class BasePersister(BasePipelineComponent):
    """Save the graph and chunk store to disk."""

    @abstractmethod
    def persist(
        self,
        graph: nx.Graph,
        chunks: list[Chunk],
        output_dir: str,
        run_metadata: RunMetadata | None = None,
    ) -> None:
        """Persist *graph* and *chunks* to *output_dir*.

        Args:
            graph: The knowledge graph built by a :class:`BaseLinker`.
            chunks: The original chunks (for building the chunk store).
            output_dir: Directory to write output files into.
            run_metadata: Configuration and stats from the pipeline run.
        """
        ...
