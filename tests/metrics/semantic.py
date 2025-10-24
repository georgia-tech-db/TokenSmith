import os
import warnings
from typing import List, Optional
from tests.metrics.base import MetricBase

class SemanticSimilarityMetric(MetricBase):
    """Semantic similarity using sentence transformers."""
    
    def __init__(self):
        self._model = None
        self._util = None
        self._available = self._initialize()
    
    @property
    def name(self) -> str:
        return "semantic"
    
    @property
    def weight(self) -> float:
        return 0.5
    
    def _initialize(self) -> bool:
        """Initialize the sentence transformer model."""
        try:
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            warnings.filterwarnings("ignore", message=".*CUDA capability.*")
            from sentence_transformers import util, SentenceTransformer
            
            self._model = SentenceTransformer('all-mpnet-base-v2')
            self._util = util
            return True
        except Exception as e:
            print(f"SemanticSimilarityMetric initialization failed: {e}")
            return False
    
    def is_available(self) -> bool:
        return self._available
    
    def calculate(self, answer: str, expected: str, keywords: Optional[List[str]] = None) -> float:
        """Calculate semantic similarity using embeddings."""
        if not self.is_available():
            return 0.0
        
        try:
            embeddings = self._model.encode([answer, expected], convert_to_tensor=True, device='cpu')
            similarity = self._util.cos_sim(embeddings[0], embeddings[1])
            return float(similarity)
        except Exception as e:
            print(f"Semantic similarity calculation failed: {e}")
            return 0.0
