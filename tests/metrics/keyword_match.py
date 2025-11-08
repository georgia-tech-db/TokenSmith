from typing import List, Optional
from tests.metrics.base import MetricBase

class KeywordMatchMetric(MetricBase):
    """Keyword matching metric."""
    
    @property
    def name(self) -> str:
        return "keyword"
    
    @property
    def weight(self) -> float:
        return 0.3
    
    def calculate(self, answer: str, expected: str, keywords: Optional[List[str]] = None) -> float:
        """Calculate keyword matching score."""
        if not keywords:
            return 0.0
        
        answer_lower = answer.lower()
        matched = sum(1 for kw in keywords if kw.lower() in answer_lower)
        return matched / len(keywords)
