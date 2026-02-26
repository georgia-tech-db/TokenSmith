from sklearn.feature_extraction.text import TfidfVectorizer

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
        self.top_n = top_n
        self.normalizer = normalizer or Normalizer()

    def extract(self, chunks: list[Chunk]) -> list[ExtractionResult]:
        texts = [c.text for c in chunks]
        if not texts:
            return []

        # Configure vectorizer. max_df and min_df are defaults for keyword extraction.
        # We use stop_words to filter out noise.
        vectorizer = TfidfVectorizer(
            max_df=0.95,
            min_df=2 if len(texts) > 1 else 1,
            stop_words="english",
            token_pattern=r"(?u)\b[a-zA-Z]{3,}\b",  # Only words with 3+ letters
        )

        try:
            tfidf_matrix = vectorizer.fit_transform(texts)
        except ValueError:
            # Handles case where vocabulary is empty after filtering
            return [ExtractionResult(chunk_id=chunk.id, nodes=[]) for chunk in chunks]

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
