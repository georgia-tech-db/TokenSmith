import logging
from typing import List, Optional

from src.knowledge_graph.base.extractor import BaseExtractor
from src.knowledge_graph.models import Chunk, ExtractionResult
from src.knowledge_graph.utils.normalizer import Normalizer

logger = logging.getLogger(__name__)


class BertopicExtractor(BaseExtractor):
    """BERTopic implementation as a knowledge graph Extractor."""

    def __init__(self, nr_topics: int = 10, normalizer: Optional[Normalizer] = None):
        self.nr_topics = nr_topics
        self.normalizer = normalizer or Normalizer()

    def extract(self, chunks: List[Chunk]) -> List[ExtractionResult]:
        raise NotImplementedError("BERTopic is not implemented yet")
        # try:
        #     from bertopic import BERTopic
        # except ImportError:
        #     logger.error(
        #         "BERTopic not installed. Please install it to use BertopicExtractor."
        #     )
        #     return [ExtractionResult(chunk_id=c.id, nodes=[]) for c in chunks]

        # texts = [c.text for c in chunks]
        # if not texts:
        #     return []

        # # BERTopic requires a certain minimum number of docs, but we'll try to run it directly
        # try:
        #     topic_model = BERTopic(
        #         nr_topics=self.nr_topics, embedding_model="all-MiniLM-L6-v2"
        #     )
        #     raw_topics, _ = topic_model.fit_transform(texts)
        #     all_topics = topic_model.get_topics()

        #     topic_id_to_idx = {}
        #     valid_topic_ids = sorted([tid for tid in all_topics.keys() if tid != -1])
        #     for idx, topic_id in enumerate(valid_topic_ids):
        #         topic_id_to_idx[topic_id] = idx

        #     ordered_extracted_topics = []
        #     for tid in valid_topic_ids:
        #         words_with_weights = all_topics[tid]
        #         ordered_extracted_topics.append([ww[0] for ww in words_with_weights])

        #     assignments = [topic_id_to_idx.get(tid, -1) for tid in raw_topics]

        # except Exception as e:
        #     logger.warning(f"BERTopic extraction failed: {e}. Returning empty chunks.")
        #     return [ExtractionResult(chunk_id=c.id, nodes=[]) for c in chunks]

        # results: List[ExtractionResult] = []
        # for i, chunk in enumerate(chunks):
        #     topic_idx = assignments[i]
        #     if topic_idx != -1 and topic_idx < len(ordered_extracted_topics):
        #         raw_nodes = ordered_extracted_topics[topic_idx]
        #     else:
        #         raw_nodes = []

        #     normalized = self.normalizer.normalize(raw_nodes)
        #     results.append(ExtractionResult(chunk_id=chunk.id, nodes=normalized))

        # return results
