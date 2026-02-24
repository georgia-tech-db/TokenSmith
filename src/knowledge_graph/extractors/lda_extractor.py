import logging
from typing import List, Optional, Tuple

from gensim.models.ldamulticore import LdaMulticore
from gensim.corpora.dictionary import Dictionary
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
import nltk

from src.knowledge_graph.base.extractor import BaseExtractor
from src.knowledge_graph.models import Chunk, ExtractionResult
from src.knowledge_graph.utils.normalizer import Normalizer

logger = logging.getLogger(__name__)

try:
    nltk.download("stopwords", quiet=True)
except Exception as e:
    logger.warning(f"Failed to download NLTK stopwords: {e}")


class LdaExtractor(BaseExtractor):
    """Gensim implementation of LDA as a knowledge graph Extractor."""

    def __init__(
        self,
        num_topics: int = 10,
        passes: int = 10,
        normalizer: Optional[Normalizer] = None,
    ):
        self.num_topics = num_topics
        self.passes = passes
        self.stop_words = set(stopwords.words("english"))
        self.normalizer = normalizer or Normalizer()

    def _preprocess(
        self, texts: List[str]
    ) -> Tuple[List[List[str]], Dictionary, List[List[Tuple[int, int]]]]:
        tokenized_docs = []
        for text in texts:
            tokens = [
                token
                for token in simple_preprocess(text)
                if token not in self.stop_words
            ]
            tokenized_docs.append(tokens)

        dictionary = Dictionary(tokenized_docs)
        corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]
        return tokenized_docs, dictionary, corpus

    def extract(self, chunks: List[Chunk]) -> List[ExtractionResult]:
        texts = [c.text for c in chunks]
        if not texts:
            return []

        _, dictionary, corpus = self._preprocess(texts)

        lda_model = LdaMulticore(
            corpus=corpus,
            id2word=dictionary,
            num_topics=self.num_topics,
            passes=self.passes,
            random_state=42,
        )

        extracted_topics = []
        for i in range(self.num_topics):
            topic_terms = lda_model.show_topic(i, topn=10)
            extracted_topics.append([term[0] for term in topic_terms])

        assignments = []
        for bow in corpus:
            probs = lda_model.get_document_topics(bow)
            if probs:
                best_topic = max(probs, key=lambda x: x[1])[0]
                assignments.append(int(best_topic))
            else:
                assignments.append(-1)

        results: List[ExtractionResult] = []
        for i, chunk in enumerate(chunks):
            topic_idx = assignments[i]
            if topic_idx != -1 and topic_idx < len(extracted_topics):
                raw_nodes = extracted_topics[topic_idx]
            else:
                raw_nodes = []

            normalized = self.normalizer.normalize(raw_nodes)
            results.append(ExtractionResult(chunk_id=chunk.id, nodes=normalized))

        return results
