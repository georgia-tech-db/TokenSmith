import spacy

from src.knowledge_graph.extractors import BaseExtractor
from src.knowledge_graph.models import Chunk, ExtractionResult
from src.knowledge_graph.utils.normalizer import Normalizer


class SpacyExtractor(BaseExtractor):
    """Extract named entities using spaCy's NER pipeline.

    Args:
        spacy_model: Name of the spaCy model to load.
        entity_types: Entity label filter (e.g. ``["PERSON", "ORG", "GPE"]``).
            ``None`` accepts all entity types.
        normalizer: Optional pre-built :class:`Normalizer`. A default one is
            created if not supplied.
    """

    def __init__(
        self,
        spacy_model: str = "en_core_web_sm",
        entity_types: list[str] | None = None,
        normalizer: Normalizer | None = None,
    ):
        self.nlp = spacy.load(spacy_model)
        self.entity_types = set(entity_types) if entity_types else None
        self.normalizer = normalizer or Normalizer(spacy_model=spacy_model)

    def extract(self, chunks: list[Chunk]) -> list[ExtractionResult]:
        results: list[ExtractionResult] = []

        for chunk in chunks:
            doc = self.nlp(chunk.text)
            raw_nodes: list[str] = []

            for ent in doc.ents:
                if self.entity_types is None or ent.label_ in self.entity_types:
                    raw_nodes.append(ent.text)

            normalized = self.normalizer.normalize(raw_nodes)
            results.append(ExtractionResult(chunk_id=chunk.id, nodes=normalized))
        print(results)
        return results
