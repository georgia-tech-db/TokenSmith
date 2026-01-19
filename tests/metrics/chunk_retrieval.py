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
        ideal_retrieved_chunks: List[int], 
        retrieved_chunks) -> float:
        # Handle None/empty inputs (e.g. agent mode doesn't provide chunk info)
        if not retrieved_chunks or not ideal_retrieved_chunks:
            return 0.0
        
        print("ideal_retrieved_chunks: ", ideal_retrieved_chunks)
        print("retrieved_chunks: ", [chunk["chunk_id"] for chunk in retrieved_chunks])
        found_chunks = [chunk["chunk_id"] for chunk in retrieved_chunks if chunk["chunk_id"] in ideal_retrieved_chunks]
        return len(found_chunks)
