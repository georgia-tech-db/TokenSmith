"""Composite extractor that merges outputs from multiple extractors."""

import time
from collections import defaultdict
from typing import List

from src.knowledge_graph.base.extractor import BaseExtractor
from src.knowledge_graph.models import Chunk, ExtractionResult
from src.knowledge_graph.utils.normalizer import Normalizer


class CompositeExtractor(BaseExtractor):
    """Merge and deduplicate outputs from multiple child extractors.

    Each child extractor runs independently on the full chunk list. Results
    for the same ``chunk_id`` are merged and deduplicated.

    Args:
        extractors: List of :class:`BaseExtractor` instances to compose.
        normalizer: Optional :class:`Normalizer` for final deduplication pass.
    """

    def __init__(
        self,
        extractors: List[BaseExtractor],
        normalizer: Normalizer | None = None,
    ):
        self.extractors = extractors
        self.normalizer = normalizer or Normalizer()

    def extract(self, chunks: List[Chunk]) -> List[ExtractionResult]:
        # Collect all nodes per chunk_id across all extractors
        merged: defaultdict[int, list] = defaultdict(list)

        total_chunks = len(chunks)
        print(
            f"Starting extraction on {total_chunks} chunks using {len(self.extractors)} extractors..."
        )

        total_start_time = time.time()

        for idx, extractor in enumerate(self.extractors, 1):
            extractor_name = extractor.__class__.__name__
            print(f"Running {extractor_name} ({idx}/{len(self.extractors)})...")

            start_time = time.time()
            for result in extractor.extract(chunks):
                merged[result.chunk_id].extend(result.nodes)
            elapsed = time.time() - start_time

            speed = elapsed / total_chunks if total_chunks > 0 else 0

            print(
                f"  -> Finished {extractor_name} in {elapsed:.2f}s ({speed:.4f}s / chunk)"
            )

        # Deduplicate per chunk via normalizer, preserve chunk ordering
        results: List[ExtractionResult] = []
        for chunk in chunks:
            raw_nodes = merged.get(chunk.id, [])
            # Nodes are already individually normalized by child extractors;
            # run normalize again to deduplicate across extractors.
            deduped = self.normalizer.normalize(raw_nodes)
            results.append(ExtractionResult(chunk_id=chunk.id, nodes=deduped))

        total_elapsed = time.time() - total_start_time
        print(f"Completed all extraction in {total_elapsed:.2f}s")
        return results
