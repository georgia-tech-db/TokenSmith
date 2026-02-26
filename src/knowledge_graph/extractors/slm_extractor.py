import logging
import json
import re

from src.knowledge_graph.extractors import BaseExtractor
from src.knowledge_graph.models import Chunk, ExtractionResult
from src.knowledge_graph.utils.normalizer import Normalizer
from src.knowledge_graph.utils.prompts import TOPIC_EXTRACTION_PROMPT

logger = logging.getLogger(__name__)


class SLMExtractor(BaseExtractor):
    """Small Language Model (SLM) implementation using llama-cpp-python as an Extractor."""

    def __init__(
        self,
        model_path: str = "models/qwen2.5-1.5b-instruct-q5_k_m.gguf",
        n_threads: int = 8,
        normalizer: Normalizer | None = None,
        prompt_template: str = TOPIC_EXTRACTION_PROMPT,
    ):
        self.model_path = model_path
        self.n_threads = n_threads
        self.normalizer = normalizer or Normalizer()
        self.prompt_template = prompt_template

    def extract(self, chunks: list[Chunk]) -> list[ExtractionResult]:
        try:
            from llama_cpp import Llama
        except ImportError:
            logger.error(
                "llama_cpp not installed. Please install it to use SlmExtractor."
            )
            return [ExtractionResult(chunk_id=c.id, nodes=[]) for c in chunks]

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
            logger.error(f"Failed to load LLM: {e}")
            return [ExtractionResult(chunk_id=c.id, nodes=[]) for c in chunks]

        limited_texts = []
        char_count = 0
        for t in texts:
            if char_count + len(t) > 12000:
                break
            limited_texts.append(t)
            char_count += len(t)

        sample_text = "\n---\n".join(limited_texts)

        prompt = self.prompt_template.format(sample_text=sample_text)
        try:
            output = llm(prompt, max_tokens=1024, temperature=0.0, stop=["<|im_end|>"])
            response_text = output["choices"][0]["text"].strip()
            logger.info(f"Raw SLM response: {response_text}")

            # Try to parse JSON from the response
            json_match = re.search(
                r"(\[[\s\n]*\[.*\][\s\n]*\])", response_text, re.DOTALL
            )
            if json_match:
                extracted_topics = json.loads(json_match.group(1))
            else:
                extracted_topics = json.loads(response_text)

            if not isinstance(extracted_topics, list) or not all(
                isinstance(t, list) for t in extracted_topics
            ):
                raise ValueError("Output is not a list of lists")
        except Exception as e:
            logger.error(f"Failed to extract/parse LLM topics: {e}")
            extracted_topics = []

        # Flatten the extracted topics from the SLM. This gives us a pool of keywords.
        # Since the SLM processes the corpus collectively (sampled) and doesn't inherently
        # return chunk-level assignments, we'll assign the union of keywords to all processed chunks
        # or we distribute them. The simplest approach that complies with BaseExtractor is to
        # treat all global extracted topics as the nodes for all chunks.

        flat_topics = set()
        for topic_list in extracted_topics:
            for word in topic_list:
                flat_topics.add(word)

        raw_nodes = list(flat_topics)
        normalized = self.normalizer.normalize(raw_nodes)

        results: list[ExtractionResult] = []
        for chunk in chunks:
            # We assign these extracted topics to the chunks. Since there's no per-chunk breakdown
            # in the prompt, everyone gets the globally extracted nodes.
            results.append(ExtractionResult(chunk_id=chunk.id, nodes=normalized))

        return results
