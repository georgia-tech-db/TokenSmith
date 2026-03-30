import spacy


class Normalizer:
    """Normalize keywords for consistent graph construction.

    Performs lowercasing, spaCy lemmatization, alias/abbreviation expansion,
    and deduplication.

    Args:
        spacy_model: Name of the spaCy model to load for lemmatization.
    """

    def __init__(self, spacy_model: str = "en_core_web_sm"):
        self.nlp = spacy.load(spacy_model, disable=["ner", "parser"])

    def _lemmatize(self, text: str) -> str:
        """Return the lemmatized form of *text*."""
        doc = self.nlp(text)
        return " ".join(token.lemma_ for token in doc)

    def normalize(self, keywords: list[str]) -> list[str]:
        """Normalize and deduplicate a list of keywords.
        Strips leading/trailing whitespace, lowercases, lemmatizes, and deduplicates.
        Keeps the first occurrence of each unique normalized keyword, preserving order.

        Args:
            keywords: Raw keyword strings.

        Returns:
            Deduplicated, normalized keywords.
        """
        result: list[str] = []
        seen: set[str] = set()  # to track duplicates after normalization
        for kw in keywords:
            normalized = kw.strip().lower()
            if not normalized:
                continue
            normalized = self._lemmatize(normalized)
            if normalized not in seen:
                seen.add(normalized)
                result.append(normalized)

        return result
