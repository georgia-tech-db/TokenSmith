from src.knowledge_graph.utils import Normalizer
import requests
import json
import logging
from typing import Any
from math import sqrt
from src.knowledge_graph.extractors.base_extractor import BaseExtractor
from src.knowledge_graph.models import Chunk, ExtractionResult

logger = logging.getLogger(__name__)


class OpenRouterExtractor(BaseExtractor):
    """Keyword extractor using OpenRouter."""

    def __init__(
        self,
        api_key: str,
        model: str,
        adaptive_top_n: bool = False,
        top_n: int = 10,
        normalizer: Normalizer | None = None,
    ):
        super().__init__()
        self.api_key = api_key
        self.model = model
        self.top_n = top_n
        self.adaptive_top_n = adaptive_top_n
        self.normalizer = normalizer or Normalizer()

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "model": self.model,
                "top_n": self.top_n,
                "adaptive_top_n": self.adaptive_top_n,
                "normalizer": self.normalizer.__class__.__name__,
            }
        )
        return config

    def extract(self, chunks: list[Chunk]) -> list[ExtractionResult]:
        results = []
        for chunk in chunks:
            try:
                top_n = (
                    int(sqrt(len(chunk.text))) if self.adaptive_top_n else self.top_n
                )
                response = requests.post(
                    url="https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    data=json.dumps(
                        {
                            "model": self.model,
                            "messages": [
                                {
                                    "role": "system",
                                    "content": f"You are a linguistic analysis expert. Analyze the provided text and identify the {top_n} most relevant and descriptive keywords or short phrases (1-3 words). Focus on terms that carry the most information density, such as technical terms, proper nouns, and central concepts. Return the result as a raw JSON list of strings. Do not include any other text or explanation in your response.",
                                },
                                {"role": "user", "content": f"Documents: {chunk.text}"},
                            ],
                        }
                    ),
                    timeout=60,
                )
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"].strip()

                # Try to parse JSON
                try:
                    keywords = json.loads(content)
                except json.JSONDecodeError:
                    # Fallback if the LLM adds markdown or something
                    import re

                    match = re.search(r"\[.*\]", content, re.DOTALL)
                    if match:
                        keywords = json.loads(match.group(0))
                    else:
                        logger.error(f"Failed to parse JSON from response: {content}")
                        keywords = []

                if not isinstance(keywords, list):
                    logger.warning(
                        f"Response for chunk {chunk.id} is not a list: {keywords}"
                    )
                    keywords = [str(keywords)] if keywords else []
                keywords = self.normalizer.normalize(keywords)
                results.append(ExtractionResult(chunk_id=chunk.id, nodes=keywords))
            except Exception as e:
                logger.error(f"Error during extraction for chunk {chunk.id}: {e}")
                results.append(ExtractionResult(chunk_id=chunk.id, nodes=[]))
        return results
