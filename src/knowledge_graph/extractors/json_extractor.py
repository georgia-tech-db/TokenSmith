
from typing import Any
from src.knowledge_graph.extractors import BaseExtractor
from src.knowledge_graph.models import Chunk, ExtractionResult
from src.knowledge_graph.utils.normalizer import Normalizer
import json

class JsonExtractor(BaseExtractor):
    """Read keywords associated to chunks in JSON.

    Args:
        input_path: Path to JSON file containing list of dicts with keys "chunk_id" and "keywords".
        normalizer: Optional pre-built :class:`Normalizer
    """

    def __init__(self, input_path: str, normalizer: Normalizer | None = None):
        super().__init__()
        self.normalizer = normalizer or Normalizer()
        self.input_path = input_path

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "input_path": self.input_path,
                "normalizer": self.normalizer.__class__.__name__,
            }
        )
        return config
    
    def extract(self, chunks: list[Chunk]) -> list[ExtractionResult]:
        with open(self.input_path, "r") as f:
            data = json.load(f)

        results: list[ExtractionResult] = []
        for entry in data:
            chunk_id = entry["chunk_id"]
            raw_nodes = entry["keywords"]
            normalized = self.normalizer.normalize(raw_nodes)
            results.append(ExtractionResult(chunk_id=chunk_id, nodes=normalized))
        return results
