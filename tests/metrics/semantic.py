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
            from sentence_transformers import util
            from src.embedder import SentenceTransformer
            
            self._model = SentenceTransformer('models/Qwen3-Embedding-4B-Q8_0.gguf')
            self._sim = util.cos_sim
            return True
        except Exception as e:
            print(f"SemanticSimilarityMetric initialization failed: {e}")
            return False
    
    def is_available(self) -> bool:
        return self._available
    
    def calculate(self, answer: str, expected: str, **kwargs) -> float:
        """
        Calculate semantic similarity using embeddings.

        This metric measures answer correctness by comparing the generated answer
        to the expected answer using sentence embeddings.

        Args:
            answer: Generated answer
            expected: Expected answer
            **kwargs: Ignored (keywords, chunks, question not used)

        Returns:
            Cosine similarity score between 0.0 and 1.0
        """
        if not self.is_available():
            return 0.0
        
        try:
            embeddings = self._model.encode([answer, expected])
            similarity = self._sim(embeddings[0], embeddings[1])
            return float(similarity)
        except Exception as e:
            print(f"Semantic similarity calculation failed: {e}")
            return 0.0
