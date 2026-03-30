"""Shared n-gram extraction and normalization helpers."""

import re

from nltk.util import ngrams

# Regex for tokenizing KG / query text.
# Matches words (including hyphenated compounds and trailing '+').
KW_PATTERN = r"\b\w+(?:\s*-\s*\w+)*\+?"

# Simpler pattern for heading text (no hyphen compounds or '+' needed).
HEADING_PATTERN = r"\b\w+\b"


def extract_ngrams(text: str, pattern: str) -> set[str]:
    """Tokenize *text*, build unigrams + bigrams + trigrams, return as a set.

    Args:
        text:       Input string to tokenize.
        pattern:    Regex pattern used to extract tokens (e.g. ``KW_PATTERN``).

    Returns:
        Set of all n-gram strings (n = 1, 2, 3).
    """
    tokens = re.findall(pattern, text)
    all_terms = list(tokens)
    for n in (2, 3):
        all_terms.extend(" ".join(gram) for gram in ngrams(tokens, n))
    return set(all_terms)
