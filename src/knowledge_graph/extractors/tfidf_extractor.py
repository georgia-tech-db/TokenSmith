from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Any
from src.knowledge_graph.extractors import BaseExtractor
from src.knowledge_graph.models import Chunk, ExtractionResult
from src.knowledge_graph.utils.normalizer import Normalizer


class TfidfExtractor(BaseExtractor):
    """Keyword extractor using TF-IDF weights to find important terms per chunk."""

    def __init__(self, top_n: int = 10, normalizer: Normalizer | None = None):
        """
        Args:
            top_n: Number of top TF-IDF words to extract per chunk.
            normalizer: Optional Normalizer for cleaning terms.
        """
        super().__init__()
        self.top_n = top_n
        self.normalizer = normalizer or Normalizer()

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {"top_n": self.top_n, "normalizer": self.normalizer.__class__.__name__}
        )
        return config

    def extract(self, chunks: list[Chunk]) -> list[ExtractionResult]:
        texts = [c.text for c in chunks]
        if not texts:
            return []

        vectorizer = TfidfVectorizer(
            stop_words="english",
            token_pattern=r"(?u)\b[a-zA-Z]{3,}\b",  # Only words with 3+ letters,
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
            normalized = self.normalizer.normalize(raw_nodes)
            results.append(ExtractionResult(chunk_id=chunk.id, nodes=normalized))

        return results
