from abc import ABC, abstractmethod
from typing import List, Optional

class MetricBase(ABC):
    """Base class for all similarity metrics."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this metric."""
        pass
    
    @property
    def weight(self) -> float:
        """Default weight for this metric in combined scoring."""
        return 1.0
    
    @abstractmethod
    def calculate(self, **kwargs) -> float:
        """
        Calculate similarity score between answer and expected.

        Metrics should accept the following common parameters via **kwargs:
            answer: Generated answer to evaluate
            expected: Expected answer or question (context-dependent)
            keywords: Optional list of keywords for keyword-based metrics
            chunks: Optional list of retrieved chunks for faithfulness/grounding metrics
            question: Original question (for LLM judge)

        Returns:
            float: Score between 0.0 and 1.0
        """
        pass
    
    def is_available(self) -> bool:
        """Check if this metric can be used (dependencies available)."""
        return True
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
