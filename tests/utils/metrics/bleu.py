from typing import List, Optional
from .base import MetricBase

class BleuScoreMetric(MetricBase):
    """BLEU score similarity metric."""
    
    @property
    def name(self) -> str:
        return "bleu"
    
    @property
    def weight(self) -> float:
        return 0.3
    
    def is_available(self) -> bool:
        """Check if NLTK is available."""
        try:
            import nltk
            return True
        except ImportError:
            return False
    
    def calculate(self, answer: str, expected: str, keywords: Optional[List[str]] = None) -> float:
        """Calculate BLEU score between answer and expected."""
        if not self.is_available():
            return 0.0
        
        try:
            from nltk.translate.bleu_score import sentence_bleu
            reference = [expected.split()]
            candidate = answer.split()
            return sentence_bleu(reference, candidate)
        except Exception as e:
            print(f"BLEU score calculation failed: {e}")
            return 0.0
