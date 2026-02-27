import logging
import json
import re
from llama_cpp import Llama

from src.knowledge_graph.extractors import BaseExtractor
from src.knowledge_graph.models import Chunk, ExtractionResult
from src.knowledge_graph.utils.normalizer import Normalizer
from src.knowledge_graph.utils.prompts import KEYWORD_EXTRACTION_PROMPT

logger = logging.getLogger(__name__)


class SLMExtractor(BaseExtractor):
    """Small Language Model implementation using llama-cpp-python as a Keyword Extractor."""

    def __init__(
        self,
        model_path: str = "models/qwen2.5-1.5b-instruct-q5_k_m.gguf",
        n_threads: int = 8,
        normalizer: Normalizer | None = None,
        prompt_template: str = KEYWORD_EXTRACTION_PROMPT,
        top_n: int = 10,
    ):
        self.model_path = model_path
        self.n_threads = n_threads
        self.normalizer = normalizer or Normalizer()
        self.prompt_template = prompt_template
        self.top_n = top_n

    def extract(self, chunks: list[Chunk]) -> list[ExtractionResult]:
        texts = [c.text for c in chunks]
        if not texts:
            return []

        try:
            llm = Llama(
                model_path=self.model_path,
                n_threads=self.n_threads,
                n_ctx=4096,
                verbose=False,
            )
        except Exception as e:
            print(f"Failed to load LLM: {e}")
            return [ExtractionResult(chunk_id=c.id, nodes=[]) for c in chunks]

        limited_texts = []
        char_count = 0
        for t in texts:
            if char_count + len(t) > 12000:
                break
            limited_texts.append(t)
            char_count += len(t)

        sample_text = "\n---\n".join(limited_texts)

        prompt = self.prompt_template.format(sample_text=sample_text, top_n=self.top_n)
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
            print(f"Failed to extract/parse LLM topics: {e}")
            extracted_topics = []

        raw_nodes = [str(t) for t in extracted_topics]
        normalized = self.normalizer.normalize(raw_nodes)

        results: list[ExtractionResult] = []
        for chunk in chunks:
            results.append(ExtractionResult(chunk_id=chunk.id, nodes=normalized))

        return results
