import difflib
from typing import List, Optional
from .base import MetricBase

class TextSimilarityMetric(MetricBase):
    """Text similarity using sequence matching."""
    
    @property
    def name(self) -> str:
        return "text"
    
    @property
    def weight(self) -> float:
        return 0.3
    
    def calculate(self, answer: str, expected: str, keywords: Optional[List[str]] = None) -> float:
        """Calculate text similarity using SequenceMatcher."""
        return difflib.SequenceMatcher(None, answer.lower(), expected.lower()).ratio()
