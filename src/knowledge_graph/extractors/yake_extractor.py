from typing import Any
import yake

from src.knowledge_graph.extractors import BaseExtractor
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
        normalizer: Normalizer | None = None,
    ):
        super().__init__()
        self.kw_extractor = yake.KeywordExtractor(
            lan=language,
            n=3,  # max n-gram size
            top=top_n,
            dedupLim=deduplicate_threshold,
        )
        self.top_n = top_n
        self.language = language
        self.deduplicate_threshold = deduplicate_threshold
        self.normalizer = normalizer or Normalizer()

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "top_n": self.top_n,
                "language": self.language,
                "deduplicate_threshold": self.deduplicate_threshold,
                "normalizer": self.normalizer.__class__.__name__,
            }
        )
        return config

    def extract(self, chunks: list[Chunk]) -> list[ExtractionResult]:
        results: list[ExtractionResult] = []

        for chunk in chunks:
            keywords = self.kw_extractor.extract_keywords(chunk.text)
            raw_nodes = [kw for kw, _score in keywords]
            normalized = self.normalizer.normalize(raw_nodes)
            results.append(ExtractionResult(chunk_id=chunk.id, nodes=normalized))

        return results
