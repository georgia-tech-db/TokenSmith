
from typing import Any
from src.knowledge_graph.extractors import BaseExtractor
from src.knowledge_graph.models import Chunk, ExtractionResult
import json


class JsonExtractor(BaseExtractor):
    """Read keywords associated to chunks in JSON.

    Args:
        input_path: Path to JSON file containing list of dicts with keys "chunk_id" and "keywords".
    """

    def __init__(self, input_path: str):
        super().__init__()
        self.input_path = input_path

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update({"input_path": self.input_path})
        return config

    def extract(self, chunks: list[Chunk]) -> list[ExtractionResult]:
        with open(self.input_path, "r") as f:
            data = json.load(f)

        return [ExtractionResult(e["chunk_id"], e["keywords"]) for e in data]
