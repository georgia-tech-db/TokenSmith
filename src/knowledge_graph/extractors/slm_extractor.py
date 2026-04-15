from typing import Any
import logging
import json
import re
from llama_cpp import Llama

from src.knowledge_graph.extractors import BaseExtractor
from src.knowledge_graph.models import Chunk, ExtractionResult
from src.knowledge_graph.prompts import KEYWORD_EXTRACTION_PROMPT

logger = logging.getLogger(__name__)


class SLMExtractor(BaseExtractor):
    """Small Language Model implementation using llama-cpp-python as a Keyword Extractor."""

    def __init__(
        self,
        model_path: str = "models/qwen2.5-1.5b-instruct-q5_k_m.gguf",
        n_threads: int = 8,
        prompt_template: str = KEYWORD_EXTRACTION_PROMPT,
        top_n: int = 10,
    ):
        super().__init__()
        self.model_path = model_path
        self.n_threads = n_threads
        self.prompt_template = prompt_template
        self.top_n = top_n

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "model_path": self.model_path,
                "n_threads": self.n_threads,
                "prompt_template": self.prompt_template,
                "top_n": self.top_n,
            }
        )
        return config

    def extract(self, chunks: list[Chunk]) -> list[ExtractionResult]:
        if not chunks:
            return []

        try:
            llm = Llama(
                model_path=self.model_path,
                n_threads=self.n_threads,
                n_ctx=4096,
                verbose=False,
            )
        except Exception as e:
            logger.error("Failed to load LLM: %s", e)
            return [ExtractionResult(chunk_id=c.id, keywords=[]) for c in chunks]

        results: list[ExtractionResult] = []
        for chunk in chunks:
            prompt = self.prompt_template.format(
                sample_text=chunk.text, top_n=self.top_n
            )
            try:
                output = llm(prompt, max_tokens=1024, temperature=0.0, stop=["<|im_end|>"])
                response_text = output["choices"][0]["text"].strip()
                json_match = re.search(r"(\[.*\])", response_text, re.DOTALL)
                if json_match:
                    extracted_topics = json.loads(json_match.group(1))
                else:
                    extracted_topics = json.loads(response_text)

                if not isinstance(extracted_topics, list):
                    raise ValueError("Output is not a list")
            except Exception as e:
                logger.warning("Failed to extract/parse LLM topics for chunk %s: %s", chunk.id, e)
                extracted_topics = []

            raw_nodes = [str(t) for t in extracted_topics]
            results.append(ExtractionResult(chunk_id=chunk.id, keywords=raw_nodes))

        return results
