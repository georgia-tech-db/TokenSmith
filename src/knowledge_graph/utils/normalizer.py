"""Node normalization and deduplication utilities."""

from typing import Dict, List, Optional

import spacy


class Normalizer:
    """Normalize node labels for consistent graph construction.

    Performs lowercasing, spaCy lemmatization, alias/abbreviation expansion,
    and deduplication.

    Args:
        alias_map: Optional mapping of abbreviations/aliases to their
            canonical forms (e.g. ``{"ai": "artificial intelligence"}``).
            Keys should be lowercase.
        spacy_model: Name of the spaCy model to load for lemmatization.
    """

    def __init__(
        self,
        alias_map: Optional[Dict[str, str]] = None,
        spacy_model: str = "en_core_web_sm",
    ):
        self.alias_map: Dict[str, str] = alias_map or {}
        self.nlp = spacy.load(spacy_model, disable=["ner", "parser"])

    def _lemmatize(self, text: str) -> str:
        """Return the lemmatized form of *text*."""
        doc = self.nlp(text)
        return " ".join(token.lemma_ for token in doc)

    def normalize(self, nodes: List[str]) -> List[str]:
        """Normalize and deduplicate a list of node labels.

        Processing order per node:
        1. Strip leading/trailing whitespace
        2. Lowercase
        3. Alias expansion (exact match on lowered string)
        4. Lemmatization via spaCy
        5. Deduplication (preserving first-seen order)

        Args:
            nodes: Raw node label strings.

        Returns:
            Deduplicated, normalized node labels.
        """
        seen: set = set()
        result: List[str] = []

        for node in nodes:
            normalized = node.strip().lower()

            # Skip empty strings
            if not normalized:
                continue

            # Alias / abbreviation expansion
            normalized = self.alias_map.get(normalized, normalized)

            # Lemmatization
            normalized = self._lemmatize(normalized)

            # Deduplication
            if normalized not in seen:
                seen.add(normalized)
                result.append(normalized)

        return result
