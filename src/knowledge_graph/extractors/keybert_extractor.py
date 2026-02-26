from keybert import KeyBERT

from src.knowledge_graph.extractor import BaseExtractor
from src.knowledge_graph.models import Chunk, ExtractionResult
from src.knowledge_graph.utils.normalizer import Normalizer


class KeyBERTExtractor(BaseExtractor):
    """Extract contextually relevant keyphrases using KeyBERT.

    Args:
        model: Sentence-transformer model name for KeyBERT.
        top_n: Maximum number of keyphrases per chunk.
        keyphrase_ngram_range: Tuple ``(min_n, max_n)`` for keyphrase
            n-gram sizes.
        normalizer: Optional pre-built :class:`Normalizer`.
    """

    def __init__(
        self,
        model: str = "all-MiniLM-L6-v2",
        top_n: int = 10,
        keyphrase_ngram_range: tuple[int, int] = (1, 2),
        normalizer: Normalizer | None = None,
    ):
        self.kw_model = KeyBERT(model=model)
        self.top_n = top_n
        self.keyphrase_ngram_range = keyphrase_ngram_range
        self.normalizer = normalizer or Normalizer()

    def extract(self, chunks: list[Chunk]) -> list[ExtractionResult]:
        results: list[ExtractionResult] = []
        for chunk in chunks:
            keywords = self.kw_model.extract_keywords(
                chunk.text,
                keyphrase_ngram_range=self.keyphrase_ngram_range,
                top_n=self.top_n,
            )
            raw_nodes = [kw for kw, _score in keywords]
            normalized = self.normalizer.normalize(raw_nodes)
            results.append(ExtractionResult(chunk_id=chunk.id, nodes=normalized))
        return results
