from typing import List, Dict, Optional, Any
from difflib import SequenceMatcher
from tests.metrics.base import MetricBase


class ChunkRetrievalMetric(MetricBase):
    """Chunk retrieval evaluation metric."""
    
    def __init__(self, similarity_threshold: float = 0.95):
        self.similarity_threshold = similarity_threshold
    
    @property
    def name(self) -> str:
        return "chunk_retrieval"
    
    def calculate(self, 
        ideal_retrieved_chunks: Optional[List[int]], 
        retrieved_chunks: Optional[List[Dict[str, Any]]]) -> float:
        if not ideal_retrieved_chunks or not retrieved_chunks:
            return 0.0
        
        found_chunks = [chunk["chunk_id"] for chunk in retrieved_chunks if chunk["chunk_id"] in ideal_retrieved_chunks]
        return len(found_chunks)
