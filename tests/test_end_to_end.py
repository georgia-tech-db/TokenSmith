import pytest
from unittest.mock import MagicMock, patch
import argparse
from typing import List, Dict, Any

# We need to mock 'src.main' imports if they try to import heavy libraries at top level
# But reading src/main.py, the imports look fine (faiss is imported, but we can't avoid that easily unless we patch sys.modules).
# Assuming environment has dependencies installed or mocks them.

from src.config import RAGConfig
from src.ranking.ranker import EnsembleRanker
from src.instrumentation.logging import RunLogger

class MockRetriever:
    def __init__(self, name: str, scores: Dict[int, float]):
        self.name = name
        self.scores = scores
    
    def get_scores(self, query: str, pool_size: int, chunks: List[str]) -> Dict[int, float]:
        # Return pre-defined scores regardless of query
        # Simulate only returning scores for valid chunk indices
        return self.scores


def test_end_to_end_pipeline_stubbed():
    """
    Test the full RAG pipeline with stubbed LLM and Vector DB.
    This ensures that the pipeline logic (retrieval aggregation, ranking, 
    prompt formatting, generation call) works efficiently without
    requiring model files or incurring inference costs.
    """
    # Import inside test to avoid import errors if environment is missing deps during collection
    from src.main import get_answer
    
    # 1. Setup Configuration

    # We must patch RAGConfig if it does validation we can't satisfy easily, 
    # but based on previous context, we can just instantiate it.
    cfg = RAGConfig(
        top_k=2,
        num_candidates=5,
        ensemble_method="linear",
        ranker_weights={"faiss": 0.5, "bm25": 0.5},
        chunk_mode="recursive_sections", # satisfy get_chunk_strategy check if any
        use_hyde=False,         # Disable HyDE to avoid extra LLM call logic complexity for this test
        disable_chunks=False,
        rerank_mode="none"      # Disable re-ranker to avoid creating another mock
    )
    
    args = argparse.Namespace(
        system_prompt_mode="baseline",
        index_prefix="test_index"
    )
    
    # 2. Setup Dummy Data
    chunks = [
        "Chunk 0: Python is a programming language.",
        "Chunk 1: The sky is blue.",
        "Chunk 2: Machine learning uses statistics.",
        "Chunk 3: Databases store data.",
        "Chunk 4: API stands for Application Programming Interface."
    ]
    sources = ["doc1", "doc1", "doc2", "doc3", "doc4"]
    
    # 3. Setup Mock Retrievers
    # We simulate that chunk 0 and chunk 2 are relevant
    # Note: get_scores expects a dict of {doc_id: score}
    faiss_scores = {0: 0.9, 2: 0.8, 1: 0.1, 3: 0.05, 4: 0.05}
    bm25_scores = {0: 0.8, 2: 0.9, 3: 0.2, 1: 0.05, 4: 0.05}
    
    retrievers = [
        MockRetriever("faiss", faiss_scores),
        MockRetriever("bm25", bm25_scores)
    ]
    
    # 4. Setup Ranker (using real EnsembleRanker)
    ranker = EnsembleRanker(
        ensemble_method="linear",
        weights={"faiss": 0.5, "bm25": 0.5}
    )
    
    artifacts = {
        "chunks": chunks,
        "sources": sources,
        "retrievers": retrievers,
        "ranker": ranker,
        "meta": [{"page_numbers": [1]} for _ in chunks]
    }
    
    # 5. Mock the Generator
    # We mock src.generator.answer directly as it's the external call from main.py
    
    # Define a generator function for the return value
    def mock_stream_generator():
        yield "This is a dummy response "
        yield "from the stubbed LLM."

    # Using patch as a context manager to mock 'src.main.answer'
    # Note: we patch where it is USED, so in src.main
    with patch("src.main.answer", return_value=mock_stream_generator()) as mock_answer_func:
        
        # Use real logger instead of mock to test logging logic coverage
        # We can patch the actual file writing method if we want to avoid disk I/O, 
        # but for max coverage we'll let it run (or just mock the save method if stricly needed)
        logger = RunLogger()
        
        # We still mock console print to avoid cluttering test output, but passed as object
        console = MagicMock()
        
        question = "What is Python?"
        
        # We use is_test_mode=True to get structured output and avoid rendering artifacts
        # This function signature matches src/main.py:get_answer
        # get_answer(question, cfg, args, logger, console, artifacts=None, golden_chunks=None, is_test_mode=False)
        
        result = get_answer(
            question=question,
            cfg=cfg,
            args=args,
            logger=logger,
            console=console,
            artifacts=artifacts,
            is_test_mode=True
        )
        
        # Unpack result (ans, chunks_info, hyde_query)
        # Assuming get_answer returns (ans, chunks_info, hyde_query) in test mode based on reading src/main.py
        ans, chunks_info, hyde_query = result
        
        # 7. Assertions
        
        # Check LLM output reconstruction
        expected_ans = "This is a dummy response from the stubbed LLM."
        assert ans == expected_ans
        
        # Check retrieval happened correctly
        # We expect top_k=2.
        # Based on MockRanker logic:
        # Chunk 0: 0.9 + 0.8 = 1.7
        # Chunk 2: 0.8 + 0.9 = 1.7
        # Chunk 1: 0.1 + 0.05 = 0.15
        # Chunk 3: 0.05 + 0.2 = 0.25
        # Top 2 should be 0 and 2 (order might depend on sort stability if equal, but they are top 2)
        
        assert len(chunks_info) == 2
        retrieved_chunk_ids = {info["chunk_id"] for info in chunks_info}
        assert 0 in retrieved_chunk_ids
        assert 2 in retrieved_chunk_ids
        
        # Check chunk content is correct in info
        for info in chunks_info:
            cid = info["chunk_id"]
            assert info["content"] == chunks[cid]
        
        # Check that answer() was called with correct context
        mock_answer_func.assert_called_once()
        
        # Inspect arguments passed to answer()
        # answer(query, chunks, model_path, max_tokens, system_prompt_mode, temperature)
        call_args = mock_answer_func.call_args
        passed_query = call_args[0][0]     # positional arg 1: query
        passed_chunks = call_args[0][1]    # positional arg 2: chunks LIST
        
        assert passed_query == question
        assert len(passed_chunks) == 2
        # Check that the passed chunks contain the text we expect
        # Note: chunks might be passed exactly as they are in the 'chunks' list
        assert any("Python is a programming language." in c for c in passed_chunks)
        assert any("Machine learning uses statistics." in c for c in passed_chunks)

