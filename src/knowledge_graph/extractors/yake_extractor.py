"""Keyword extraction via YAKE (unsupervised, no GPU needed)."""

from typing import List, Optional

import yake

from src.knowledge_graph.base.extractor import BaseExtractor
from src.knowledge_graph.models import Chunk, ExtractionResult
from src.knowledge_graph.utils.normalizer import Normalizer


class YakeExtractor(BaseExtractor):
    """Extract top-N keywords using the YAKE algorithm.

    Args:
        top_n: Maximum number of keywords to extract per chunk.
        language: Language code for YAKE (default ``"en"``).
        deduplicate_threshold: YAKE deduplication threshold (0–1). Higher
            values allow more similar keywords.
        normalizer: Optional pre-built :class:`Normalizer`.
    """

    def __init__(
        self,
        top_n: int = 10,
        language: str = "en",
        deduplicate_threshold: float = 0.9,
        normalizer: Optional[Normalizer] = None,
    ):
        self.kw_extractor = yake.KeywordExtractor(
            lan=language,
            n=3,  # max n-gram size
            top=top_n,
            dedupLim=deduplicate_threshold,
        )
        self.normalizer = normalizer or Normalizer()

    def extract(self, chunks: List[Chunk]) -> List[ExtractionResult]:
        results: List[ExtractionResult] = []

        for chunk in chunks:
            keywords = self.kw_extractor.extract_keywords(chunk.text)
            raw_nodes = [kw for kw, _score in keywords]
            normalized = self.normalizer.normalize(raw_nodes)
            results.append(ExtractionResult(chunk_id=chunk.id, nodes=normalized))

        return results
