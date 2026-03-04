from time import time
import networkx as nx

from src.knowledge_graph.dividers import BaseDivider
from src.knowledge_graph.extractors import BaseExtractor
from src.knowledge_graph.linkers import BaseLinker
from src.knowledge_graph.persisters import BasePersister
from src.knowledge_graph.models import Chunk, RunMetadata


class Pipeline:
    """Orchestrates the knowledge graph construction pipeline.

    Args:
        divider: Splits raw text into chunks. Optional, can give chunks directly in run().
        extractor: Extracts node labels from chunks.
        linker: Builds a graph from extraction results.
        persister: Saves the graph and chunk store to disk.
    """

    def __init__(
        self,
        extractor: BaseExtractor,
        linker: BaseLinker,
        persister: BasePersister,
        divider: BaseDivider | None = None,
        verbose: bool = False,
    ):
        self.divider = divider
        self.extractor = extractor
        self.linker = linker
        self.persister = persister
        self.verbose = verbose

    def run(
        self,
        output_dir: str,
        text: str | None = None,
        chunks: list[Chunk] | None = None,
    ) -> nx.Graph:
        """Execute the full pipeline.

        Args:
            text: Raw corpus text.
            chunks: Pre-chunked text. Optional.
            output_dir: Directory to write output files into.

        Returns:
            The constructed :class:`networkx.Graph`.
        """
        if self.divider:
            if text is None:
                raise ValueError("Text must be provided")
            self.log_msg("Dividing text...")
            t0 = time()
            chunks = self.divider.divide(text)
            t1 = time()
            self.log_msg(f"  {len(chunks)} chunks created in {t1 - t0:.2f} seconds")
        else:
            if chunks is None:
                raise ValueError("Chunks must be provided")
            self.log_msg(f"  {len(chunks)} chunks loaded from memory")
        self.log_msg("Extracting nodes...")
        t0 = time()
        extractions = self.extractor.extract(chunks)
        t1 = time()
        self.log_msg(
            f"  {len(extractions)} extractions created in {t1 - t0:.2f} seconds"
        )
        self.log_msg("Linking co-occurrences...")
        t0 = time()
        graph = self.linker.link(extractions)
        t1 = time()
        self.log_msg(
            f"  {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges in {t1 - t0:.2f} seconds"
        )
        self.log_msg("Persisting graph...")
        t0 = time()

        # Collect run metadata
        run_config = {
            "extractor": self.extractor.get_config(),
            "linker": self.linker.get_config(),
            "persister": self.persister.get_config(),
        }
        if self.divider:
            run_config["divider"] = self.divider.get_config()

        run_stats = {
            "extractor": self.extractor.metadata,
            "linker": self.linker.metadata,
            "persister": self.persister.metadata,
        }
        if self.divider:
            run_stats["divider"] = self.divider.metadata

        run_metadata = RunMetadata(config=run_config, statistics=run_stats)

        self.persister.persist(graph, chunks, output_dir, run_metadata=run_metadata)
        t1 = time()
        self.log_msg(f"  Graph persisted in {t1 - t0:.2f} seconds")
        # Quick stats
        self.log_msg("═" * 50)
        self.log_msg(f"  Chunks:  {len(chunks)}")
        self.log_msg(f"  Nodes:   {graph.number_of_nodes()}")
        self.log_msg(f"  Edges:   {graph.number_of_edges()}")
        self.log_msg(f"  Output:  {output_dir}")
        self.log_msg("═" * 50)
        return graph

    def log_msg(self, msg):
        if self.verbose:
            print(msg)
