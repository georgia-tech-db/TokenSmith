from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Any
from src.knowledge_graph.extractors import BaseExtractor
from src.knowledge_graph.models import Chunk, ExtractionResult


class TfidfExtractor(BaseExtractor):
    """Keyword extractor using TF-IDF weights to find important terms per chunk."""

    def __init__(self, top_n: int = 10):
        """
        Args:
            top_n: Number of top TF-IDF words to extract per chunk.
        """
        super().__init__()
        self.top_n = top_n

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update({"top_n": self.top_n})
        return config

    def extract(self, chunks: list[Chunk]) -> list[ExtractionResult]:
        texts = [c.text for c in chunks]
        if not texts:
            return []

        vectorizer = TfidfVectorizer(
            stop_words="english",
            # Only words with 3+ letters,
            token_pattern=r"(?u)\b[a-zA-Z]{3,}\b",
            ngram_range=(1, 2),
        )

        tfidf_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()
        results: list[ExtractionResult] = []
        for i, chunk in enumerate(chunks):
            # Get the row for this chunk
            row = tfidf_matrix.getrow(i).toarray().flatten()

            # Get indices of terms with non-zero scores, sorted by score descending
            top_indices = row.argsort()[::-1][: self.top_n]

            raw_nodes = []
            for idx in top_indices:
                if row[idx] > 0:
                    raw_nodes.append(feature_names[idx])
            results.append(ExtractionResult(
                chunk_id=chunk.id, keywords=raw_nodes))

        return results
