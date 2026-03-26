import logging
import time
from collections import defaultdict
from typing import Any

from src.knowledge_graph.models import Chunk, ExtractionResult
from src.knowledge_graph.extractors import BaseExtractor


logger = logging.getLogger(__name__)


class CompositeExtractor(BaseExtractor):
    """Merge and deduplicate outputs from multiple child extractors.

    Each child extractor runs independently on the full chunk list. Results
    for the same ``chunk_id`` are merged and deduplicated.

    Args:
        extractors: List of :class:`BaseExtractor` instances to compose.
    """

    def __init__(self, extractors: list[BaseExtractor]):
        super().__init__()
        self.extractors = extractors

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update({"extractors": [e.get_config()
                      for e in self.extractors]})
        return config

    def extract(self, chunks: list[Chunk]) -> list[ExtractionResult]:
        # Collect all nodes per chunk_id across all extractors
        merged: defaultdict[int, set] = defaultdict(set)

        total_chunks = len(chunks)
        logger.info(
            "Starting extraction on %d chunks using %d extractors...",
            total_chunks, len(self.extractors),
        )

        total_start_time = time.time()

        for idx, extractor in enumerate(self.extractors, 1):
            extractor_name = extractor.__class__.__name__
            logger.info("Running %s (%d/%d)...", extractor_name,
                        idx, len(self.extractors))

            start_time = time.time()
            for result in extractor.extract(chunks):
                merged[result.chunk_id].update(result.keywords)
            elapsed = time.time() - start_time

            speed = elapsed / total_chunks if total_chunks > 0 else 0
            logger.info("  -> Finished %s in %.2fs (%.4fs / chunk)",
                        extractor_name, elapsed, speed)

        # Merge extraction results
        results: list[ExtractionResult] = []
        for chunk in chunks:
            raw_keywords = merged.get(chunk.id, [])
            results.append(ExtractionResult(
                chunk_id=chunk.id, keywords=list(raw_keywords)))

        total_elapsed = time.time() - total_start_time
        logger.info("Completed all extraction in %.2fs", total_elapsed)
        return results
