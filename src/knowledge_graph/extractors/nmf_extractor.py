from typing import List, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

from src.knowledge_graph.base.extractor import BaseExtractor
from src.knowledge_graph.models import Chunk, ExtractionResult
from src.knowledge_graph.utils.normalizer import Normalizer


class NmfExtractor(BaseExtractor):
    """Scikit-Learn implementation of NMF as a knowledge graph Extractor."""

    def __init__(self, n_components: int = 10, normalizer: Optional[Normalizer] = None):
        self.n_components = n_components
        self.normalizer = normalizer or Normalizer()

    def extract(self, chunks: List[Chunk]) -> List[ExtractionResult]:
        texts = [c.text for c in chunks]
        if not texts:
            return []

        vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words="english")

        # If we have only 1 chunk, min_df=2 will fail. Need to handle small corpuses nicely.
        if len(texts) < 2:
            vectorizer = TfidfVectorizer(max_df=1.0, stop_words="english")

        try:
            tfidf = vectorizer.fit_transform(texts)
        except ValueError:
            # Handles empty vocabulary
            return [ExtractionResult(chunk_id=chunk.id, nodes=[]) for chunk in chunks]

        # Ensure n_components <= number of terms and docs
        actual_components = min(self.n_components, tfidf.shape[0], tfidf.shape[1])
        if actual_components == 0:
            return [ExtractionResult(chunk_id=chunk.id, nodes=[]) for chunk in chunks]

        nmf = NMF(
            n_components=actual_components, random_state=42, init="nndsvd", max_iter=500
        ).fit(tfidf)

        feature_names = vectorizer.get_feature_names_out()
        extracted_topics = []
        for topic in nmf.components_:
            # Get indices of top 10 words
            top_indices = topic.argsort()[:-11:-1]
            top_words = [feature_names[i] for i in top_indices]
            extracted_topics.append(top_words)

        # Assignments
        W = nmf.transform(tfidf)
        assignments = W.argmax(axis=1).tolist()

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
