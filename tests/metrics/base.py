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
    def calculate(self, answer: str, expected: str, keywords: Optional[List[str]] = None) -> float:
        """Calculate similarity score between answer and expected."""
        pass
    
    def is_available(self) -> bool:
        """Check if this metric can be used (dependencies available)."""
        return True
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
