import spacy
import pytextrank  # noqa: F401
from typing import Any
from src.knowledge_graph.extractors import BaseExtractor
from src.knowledge_graph.models import Chunk, ExtractionResult
from src.knowledge_graph.utils.normalizer import Normalizer


class TextRankExtractor(BaseExtractor):
    """Extract named entities using spaCy's TextRank pipeline.

    Args:
        spacy_model: Name of the spaCy model to load.
        normalizer: Optional pre-built :class:`Normalizer`. A default one is
            created if not supplied.
        top_n: Maximum number of keywords to extract.
    """

    def __init__(
        self,
        spacy_model: str = "en_core_web_sm",
        normalizer: Normalizer | None = None,
        top_n: int = 10,
    ):
        super().__init__()
        self.spacy_model = spacy_model
        self.nlp = spacy.load(spacy_model)
        self.nlp.add_pipe("textrank")
        self.normalizer = normalizer or Normalizer(spacy_model=spacy_model)
        self.top_n = top_n

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "spacy_model": self.spacy_model,
                "top_n": self.top_n,
                "normalizer": self.normalizer.__class__.__name__,
            }
        )
        return config

    def extract(self, chunks: list[Chunk]) -> list[ExtractionResult]:
        results: list[ExtractionResult] = []
        for chunk in chunks:
            doc = self.nlp(chunk.text)
            raw_nodes: list[str] = []

            for phrase in doc._.phrases[: self.top_n]:
                raw_nodes.append(phrase.text)

            normalized = self.normalizer.normalize(raw_nodes)
            results.append(ExtractionResult(chunk_id=chunk.id, nodes=normalized))
        return results
