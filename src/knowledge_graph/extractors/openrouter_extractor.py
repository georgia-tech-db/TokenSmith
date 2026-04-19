import json
import logging
import re
from math import sqrt
from typing import Any

from src.knowledge_graph.extractors.base_extractor import BaseExtractor
from src.knowledge_graph.models import Chunk, ExtractionResult
from src.knowledge_graph.openrouter_client import OpenRouterClient
from src.knowledge_graph.prompts import OPENROUTER_KEYWORD_EXTRACTION_PROMPT

logger = logging.getLogger(__name__)


class OpenRouterExtractor(BaseExtractor):
    """Keyword extractor using OpenRouter."""

    def __init__(
        self,
        api_key: str,
        model: str,
        adaptive_top_n: bool = False,
        top_n: int = 10,
        retries: int = 1,
    ):
        super().__init__()
        self.model = model
        self.top_n = top_n
        self.adaptive_top_n = adaptive_top_n
        self._client = OpenRouterClient(api_key, retries=retries)

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "model": self.model,
                "top_n": self.top_n,
                "adaptive_top_n": self.adaptive_top_n,
            }
        )
        return config

    def _parse_keywords(self, content: str) -> list[str]:
        """Parse a JSON list of keywords from the LLM response string.

        Raises:
            ValueError: When the content cannot be parsed as a list of strings.
        """
        try:
            keywords = json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\[.*\]", content, re.DOTALL)
            if match:
                keywords = json.loads(match.group(0))
            else:
                raise ValueError(f"Cannot parse JSON list from response: {content!r}")

        if not isinstance(keywords, list):
            raise ValueError(f"Response is not a list: {keywords!r}")

        return keywords

    def extract(self, chunks: list[Chunk]) -> list[ExtractionResult]:
        results = []
        for chunk in chunks:
            top_n = int(sqrt(len(chunk.text))) if self.adaptive_top_n else self.top_n
            messages = [
                {
                    "role": "system",
                    "content": OPENROUTER_KEYWORD_EXTRACTION_PROMPT.format(top_n=top_n),
                },
                {"role": "user", "content": f"Documents: {chunk.text}"},
            ]
            keywords: list[str] = []
            try:
                content = self._client.chat(model=self.model, messages=messages)
                keywords = self._parse_keywords(content)
            except Exception as e:
                logger.error("Chunk %s: all attempts failed — %s", chunk.id, e)
            results.append(ExtractionResult(chunk_id=chunk.id, keywords=keywords))
        return results
