from keybert import KeyBERT
from typing import Any
from src.knowledge_graph.extractors import BaseExtractor
from src.knowledge_graph.models import Chunk, ExtractionResult


class KeyBERTExtractor(BaseExtractor):
    """Extract contextually relevant keyphrases using KeyBERT.

    Args:
        model: Sentence-transformer model name for KeyBERT.
        top_n: Maximum number of keyphrases per chunk.
        keyphrase_ngram_range: Tuple ``(min_n, max_n)`` for keyphrase
            n-gram sizes.
    """

    def __init__(
        self,
        model: str = "all-MiniLM-L6-v2",
        top_n: int = 10,
        keyphrase_ngram_range: tuple[int, int] = (1, 2),
    ):
        super().__init__()
        self.model_name = model
        self.kw_model = KeyBERT(model=model)
        self.top_n = top_n
        self.keyphrase_ngram_range = keyphrase_ngram_range

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "model": self.model_name,
                "top_n": self.top_n,
                "keyphrase_ngram_range": self.keyphrase_ngram_range,
            }
        )
        return config

    def extract(self, chunks: list[Chunk]) -> list[ExtractionResult]:
        results: list[ExtractionResult] = []
        for chunk in chunks:
            keywords = self.kw_model.extract_keywords(
                chunk.text,
                keyphrase_ngram_range=self.keyphrase_ngram_range,
                top_n=self.top_n,
            )
            raw_nodes = [kw for kw, _ in keywords]
            results.append(ExtractionResult(chunk_id=chunk.id, keywords=raw_nodes))
        return results
