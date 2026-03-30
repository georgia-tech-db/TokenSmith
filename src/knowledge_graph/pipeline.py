import logging
from time import time

import networkx as nx

logger = logging.getLogger(__name__)

from src.knowledge_graph.canonicalizer import Canonicalizer
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
        canonicalizer: Optional semantic canonicalization step between extraction and linking.
    """

    def __init__(
        self,
        extractor: BaseExtractor,
        linker: BaseLinker,
        persister: BasePersister,
        divider: BaseDivider | None = None,
        canonicalizer: Canonicalizer | None = None,
    ):
        self.divider = divider
        self.extractor = extractor
        self.linker = linker
        self.persister = persister
        self.canonicalizer = canonicalizer

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
            logger.info("Dividing text...")
            t0 = time()
            chunks = self.divider.divide(text)
            t1 = time()
            logger.info(f"  {len(chunks)} chunks created in {t1 - t0:.2f} seconds")
        else:
            if chunks is None:
                raise ValueError("Chunks must be provided")
            logger.info(f"  {len(chunks)} chunks loaded from memory")
        logger.info("Extracting nodes...")
        t0 = time()
        extractions = self.extractor.extract(chunks)
        t1 = time()
        logger.info(
            f"  {len(extractions)} extractions created in {t1 - t0:.2f} seconds"
        )

        canon_result = None
        if self.canonicalizer:
            logger.info("Canonicalizing keywords...")
            t0 = time()
            extractions, canon_result = self.canonicalizer.canonicalize(extractions)
            t1 = time()
            s = canon_result.stats
            logger.info(
                f"  {s['keywords_after_stage1']} → {s['canonical_keywords_final']} keywords, "
                f"{s['merges_performed']} merges, {s['llm_calls']} LLM calls "
                f"in {t1 - t0:.2f} seconds"
            )

        logger.info("Linking co-occurrences...")
        t0 = time()
        graph = self.linker.link(extractions)
        t1 = time()
        logger.info(
            f"  {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges in {t1 - t0:.2f} seconds"
        )
        logger.info("Persisting graph...")
        t0 = time()

        # Collect run metadata
        run_config = {
            "extractor": self.extractor.get_config(),
            "linker": self.linker.get_config(),
            "persister": self.persister.get_config(),
        }
        if self.divider:
            run_config["divider"] = self.divider.get_config()
        if self.canonicalizer:
            run_config["canonicalizer"] = self.canonicalizer.get_config()

        run_stats = {
            "extractor": self.extractor.metadata,
            "linker": self.linker.metadata,
            "persister": self.persister.metadata,
        }
        if self.divider:
            run_stats["divider"] = self.divider.metadata
        if canon_result:
            run_stats["canonicalization"] = canon_result.stats

        run_metadata = RunMetadata(config=run_config, statistics=run_stats)

        self.persister.persist(
            graph, chunks, output_dir,
            run_metadata=run_metadata,
            canonicalization_result=canon_result,
        )
        t1 = time()
        logger.info(f"  Graph persisted in {t1 - t0:.2f} seconds")
        # Quick stats
        logger.info("═" * 50)
        logger.info(f"  Chunks:  {len(chunks)}")
        logger.info(f"  Nodes:   {graph.number_of_nodes()}")
        logger.info(f"  Edges:   {graph.number_of_edges()}")
        logger.info(f"  Output:  {output_dir}")
        logger.info("═" * 50)
        return graph

