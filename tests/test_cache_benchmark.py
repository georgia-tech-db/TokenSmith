
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from src.config import RAGConfig
from src.cache import (
    SEMANTIC_CACHE,
    semantic_cache_store,
    semantic_cache_lookup,
    compute_question_embedding,
    make_cache_config_key,
    normalize_question
)

import pytest
import numpy as np
import yaml
from pathlib import Path
from unittest.mock import MagicMock, patch
from src.config import RAGConfig
from src.cache import (
    SEMANTIC_CACHE,
    semantic_cache_store,
    semantic_cache_lookup,
    compute_question_embedding,
    make_cache_config_key,
    normalize_question
)

def load_benchmark_data():
    """Load benchmark questions from YAML."""
    yaml_path = Path(__file__).parent / "cache_benchmark.yaml"
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)

BENCHMARK_DATA = load_benchmark_data()

@pytest.fixture
def mock_config():
    """Create a mock RAGConfig for testing."""
    config = MagicMock(spec=RAGConfig)
    config.gen_model = "mock-model"
    config.embed_model = "models/Qwen3-Embedding-4B-Q5_K_M.gguf" # Use local available model
    config.top_k = 5
    config.system_prompt_mode = "baseline"
    config.ensemble_method = "rrf"
    config.ranker_weights = {"faiss": 0.5, "bm25": 0.5}
    config.use_hyde = False
    config.use_indexed_chunks = False
    config.disable_chunks = False
    config.use_golden_chunks = False
    return config

def test_cache_benchmark_comprehensive(mock_config):
    """
    Test 30 questions with ~10-15 variations each to verify semantic cache.
    """
    # 1. Clear cache
    SEMANTIC_CACHE.clear()
    
    # Setup
    args = MagicMock()
    args.model_path = None
    args.system_prompt_mode = None
    args.index_prefix = "test_index"
    
    cache_key = make_cache_config_key(mock_config, args, None)
    embed_model_name = mock_config.embed_model
    
    total_hits = 0
    total_variations = 0
    failures = []

    print(f"\n{'='*60}")
    print(f"  RUNNING COMPREHENSIVE CACHE BENCHMARK")
    print(f"  Model: {embed_model_name}")
    print(f"{'='*60}")

    for entry in BENCHMARK_DATA:
        question_id = entry["id"]
        main_question = entry["question"]
        variations = entry["variations"]
        
        print(f"\n[{question_id}] Seeding: '{main_question}'")
        
        # 2. Seed the main question
        normalized_main = normalize_question(main_question)
        embedding_main = compute_question_embedding(normalized_main, [], embed_model_name)
        assert embedding_main is not None, f"Failed to compute embedding for {question_id}"
        
        payload = {
            "answer": f"Cached answer for {question_id}",
            "chunks_info": [],
            "hyde_query": None,
            "chunk_indices": []
        }
        
        semantic_cache_store(cache_key, normalized_main, embedding_main, payload)
        
        # 3. Test variations
        entry_hits = 0
        
        for var_q in variations:
            normalized_var = normalize_question(var_q)
            embedding_var = compute_question_embedding(normalized_var, [], embed_model_name)
            
            result = semantic_cache_lookup(cache_key, embedding_var, normalized_var)
            
            if result:
                print(f"  ✅ Hit: '{var_q}'")
                entry_hits += 1
            else:
                failures.append(f"[{question_id}] Missed: '{var_q}'")
                print(f"  ❌ Missed: '{var_q}'")
        
        total_hits += entry_hits
        total_variations += len(variations)
        print(f"  ✅ Hits: {entry_hits}/{len(variations)}")

    # Summary
    hit_rate = total_hits / total_variations if total_variations > 0 else 0
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"  Total Variations: {total_variations}")
    print(f"  Total Hits: {total_hits}")
    print(f"  Hit Rate: {hit_rate:.2%}")
    print(f"{'='*60}")
    
    # Assert acceptable hit rate (e.g., >80% for semantic similarity)
    # Given we are using a good embedding model, it should be high.
    assert hit_rate >= 0.80, f"Cache hit rate {hit_rate:.2%} is below 80%. Failures: {failures[:10]}..."
