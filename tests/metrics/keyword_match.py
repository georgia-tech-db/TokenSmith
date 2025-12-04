import re
from typing import List, Optional
from tests.metrics.base import MetricBase

class KeywordMatchMetric(MetricBase):
    """
    Keyword matching metric with fuzzy matching support.

    Uses both exact substring matching and fuzzy matching to account for
    variations in word forms (plurals, verb tenses, etc.).
    """

    def __init__(self, fuzzy_threshold: int = 85):
        """
        Initialize keyword matcher.

        Args:
            fuzzy_threshold: Minimum fuzzy match score (0-100) to consider a match
        """
        self.fuzzy_threshold = fuzzy_threshold

    @property
    def name(self) -> str:
        return "keyword"
    
    @property
    def weight(self) -> float:
        return 0.3

    def _normalize_word(self, word: str) -> str:
        """Normalize a word by removing punctuation and converting to lowercase."""
        return re.sub(r'[^\w\s]', '', word.lower())

    def _exact_match(self, keyword: str, answer: str) -> bool:
        """Check if keyword appears in answer (case-insensitive, substring match)."""
        kw_lower = keyword.lower()
        answer_lower = answer.lower()
        return kw_lower in answer_lower

    def _fuzzy_match(self, keyword: str, answer: str) -> bool:
        """Check if keyword matches any word in answer using fuzzy matching."""

        from rapidfuzz import fuzz

        kw_normalized = self._normalize_word(keyword)
        # Split answer into words and normalize them
        answer_words = [self._normalize_word(word) for word in answer.split()]

        # Check if keyword matches any word with fuzzy threshold
        for word in answer_words:
            if fuzz.ratio(kw_normalized, word) >= self.fuzzy_threshold:
                return True

        return False

    def calculate(self, answer: str, keywords: Optional[List[str]] = None, **kwargs) -> float:
        """
        Calculate keyword matching score with fuzzy matching.

        Args:
            answer: Generated answer to check for keywords
            keywords: List of keywords to look for
            **kwargs: Ignored (expected, chunks, question not used)

        Returns:
            Ratio of matched keywords to total keywords (0.0 to 1.0)
        """
        if not keywords:
            return 0.0

        matched = 0
        for keyword in keywords:
            # Try exact match first (faster)
            if self._exact_match(keyword, answer):
                matched += 1
            # Fall back to fuzzy match if available
            elif self._fuzzy_match(keyword, answer):
                matched += 1

        return matched / len(keywords)
