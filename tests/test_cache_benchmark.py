import pytest
import numpy as np
import yaml
import os
from pathlib import Path
from unittest.mock import MagicMock
from src.config import RAGConfig
from src.cache import get_cache


# -----------------------------
# Data loading
# -----------------------------

def load_benchmark_data():
    """Load benchmark questions from YAML."""
    yaml_path = Path(__file__).parent / "cache_benchmark.yaml"
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


BENCHMARK_DATA = load_benchmark_data()
pytestmark = pytest.mark.skipif(
    os.environ.get("TOKENSMITH_RUN_CACHE_BENCHMARK") != "1",
    reason="semantic cache benchmark requires TOKENSMITH_RUN_CACHE_BENCHMARK=1",
)


# -----------------------------
# Fixtures
# -----------------------------

@pytest.fixture
def mock_config():
    """Create a mock RAGConfig for testing."""
    config = MagicMock(spec=RAGConfig)
    config.gen_model = "mock-model"
    config.embed_model = "models/embedders/Qwen3-Embedding-4B-Q5_K_M.gguf"
    config.top_k = 5
    config.system_prompt_mode = "baseline"
    config.ensemble_method = "rrf"
    config.ranker_weights = {"faiss": 0.5, "bm25": 0.5}
    config.use_hyde = False
    config.use_indexed_chunks = False
    config.disable_chunks = False
    config.use_golden_chunks = False
    config.semantic_cache_enabled = True
    config.semantic_cache_bi_encoder_threshold = 0.90
    config.semantic_cache_cross_encoder_threshold = 0.99
    return config


# -----------------------------
# Helpers
# -----------------------------

def print_separator():
    print("*" * 65)


def print_question_block(question_id, main_question, var_results, adversarial_results):
    """
    Print a formatted block for a single question with tick/X results
    for both variations and adversarial_queries.

    var_results:   list of (question_str, hit: bool)
    adversarial_results: list of (question_str, hit: bool)
    """
    print_separator()
    print(f"  [{question_id}] {main_question}")
    print()

    print("  Variations (hits are good ✅):")
    for q, hit in var_results:
        symbol = "✅" if hit else "❌"
        print(f"    {symbol}  {q}")

    print()
    print("  Adversarial Queries (hits are bad ⚠️):")
    for q, hit in adversarial_results:
        symbol = "⚠️ " if hit else "✅"
        print(f"    {symbol}  {q}")

    var_hits = sum(1 for _, h in var_results if h)
    adversarial_hits = sum(1 for _, h in adversarial_results if h)
    print()
    print(f"  Accuracy : {var_hits}/{len(var_results)} variations matched")
    print(f"  False Positives: {adversarial_hits}/{len(adversarial_results)} adversarial queries falsely matched")


# -----------------------------
# Test
# -----------------------------

def test_cache_benchmark_comprehensive(mock_config):
    """
    Benchmark the semantic cache against 30 questions, each with:
      - 5 genuine paraphrase variations  (hits expected)
      - 5 adversarial queries               (hits NOT expected)

    Scores:
      Accuracy rate  — fraction of genuine variations that got a cache hit.
                     Higher is better. Target >= 60%.
      False Positives rate — fraction of adversarial queries that falsely got a cache hit.
                     Lower is better. Target <= 0%.
    """
    
    cache = get_cache(mock_config)
    cache.clear()

    args = MagicMock()
    args.model_path = None
    args.system_prompt_mode = None
    args.index_prefix = "test_index"

    cache_key = cache.make_config_key(mock_config, args, None)
    embed_model_name = mock_config.embed_model

    total_var_hits = 0
    total_variations = 0
    total_adversarial_hits = 0
    total_adversarial_queries = 0

    accuracy_failures = []
    false_positive_failures = []

    print(f"\n{'*' * 65}")
    print(f"  SEMANTIC CACHE BENCHMARK")
    print(f"  Embedding model : {embed_model_name}")
    print(f"  Questions       : {len(BENCHMARK_DATA)}")
    print(f"  Variations each : 5 genuine  +  5 adversarial")
    print(f"{'*' * 65}")

    for entry in BENCHMARK_DATA:
        question_id    = entry["id"]
        main_question  = entry["question"]
        variations     = entry.get("variations", [])
        adversarial_vars     = entry.get("adversarial_queries", [])

        # --- Seed the cache with the canonical question ---
        normalized_main = cache.normalize_question(main_question)
        try:
            embedding_main = cache.compute_embedding(normalized_main, [], embed_model_name)
        except ValueError as exc:
            pytest.skip(f"Semantic cache benchmark requires a loadable embedding model: {exc}")
        if embedding_main is None:
            pytest.skip("Semantic cache benchmark requires embedding support.")

        payload = {
            "answer":        f"Cached answer for {question_id}",
            "chunks_info":   [],
            "hyde_query":    None,
            "chunk_indices": [],
        }
        cache.store(cache_key, normalized_main, embedding_main, payload)

        # --- Test genuine variations ---
        var_results = []
        for var_q in variations:
            normalized_var = cache.normalize_question(var_q)
            embedding_var  = cache.compute_embedding(normalized_var, [], embed_model_name)
            hit = cache.lookup(cache_key, embedding_var, normalized_var) is not None
            var_results.append((var_q, hit))
            if not hit:
                accuracy_failures.append(f"[{question_id}] Missed variation : '{var_q}'")

        # --- Test adversarial queries ---
        adversarial_results = []
        for adversarial_q in adversarial_vars:
            normalized_adversarial = cache.normalize_question(adversarial_q)
            embedding_adversarial  = cache.compute_embedding(normalized_adversarial, [], embed_model_name)
            
            payload_hit = cache.lookup(cache_key, embedding_adversarial, normalized_adversarial)
            # A false hit is when the cache incorrectly returns the CURRENT question's answer 
            # for a query that is semantically different.
            hit = payload_hit is not None and payload_hit.get("answer") == f"Cached answer for {question_id}"
            adversarial_results.append((adversarial_q, hit))
            if hit:
                false_positive_failures.append(f"[{question_id}] False hit on adversarial: '{adversarial_q}'")

        # --- Accumulate totals ---
        total_var_hits   += sum(1 for _, h in var_results   if h)
        total_variations += len(var_results)
        total_adversarial_hits += sum(1 for _, h in adversarial_results if h)
        total_adversarial_queries     += len(adversarial_results)

        # --- Print per-question block ---
        print_question_block(question_id, main_question, var_results, adversarial_results)

    # --- Final summary ---
    accuracy_rate  = total_var_hits   / total_variations if total_variations > 0 else 0.0
    false_positive_rate = total_adversarial_hits / total_adversarial_queries     if total_adversarial_queries     > 0 else 0.0

    print_separator()
    print()
    print(f"  {'FINAL BENCHMARK RESULTS':^61}")
    print()
    print(f"  {'Metric':<35} {'Score':>10}   {'Count'}")
    print(f"  {'-'*60}")
    print(f"  {'Accuracy Rate  (higher is better)':<35} {accuracy_rate:>9.1%}   {total_var_hits}/{total_variations}")
    print(f"  {'False Positive Rate (lower  is better)':<35} {false_positive_rate:>9.1%}   {total_adversarial_hits}/{total_adversarial_queries}")
    print()
    print("  What these scores mean:")
    print()
    print("  Accuracy Rate — measures how often the cache correctly")
    print("  recognises a genuine paraphrase of a cached question.")
    print("  A high accuracy rate means users asking the same thing")
    print("  in different words will get a fast cached response.")
    print("  Target: >= 60%")
    print()
    print("  False Positive Rate — measures how often the cache is fooled")
    print("  into returning an answer for a semantically or")
    print("  syntactically similar but DIFFERENT question. A false")
    print("  hit here means a user gets the wrong cached answer.")
    print("  Lower is strictly better. Target: <= 0%")

    if accuracy_failures:
        print()
        print(f"  Accuracy misses ({len(accuracy_failures)} total, showing first 10):")
        for msg in accuracy_failures[:10]:
            print(f"    ❌ {msg}")

    if false_positive_failures:
        print()
        print(f"  False Positive hits ({len(false_positive_failures)} total, showing first 10):")
        for msg in false_positive_failures[:10]:
            print(f"    ⚠️  {msg}")

    print()
    print_separator()

    # --- Assertions ---
    assert accuracy_rate >= 0.80, (
        f"Accuracy rate {accuracy_rate:.1%} is below the 80% target. "
        f"The cache is missing too many genuine paraphrases.\n"
        f"First 10 misses: {accuracy_failures[:10]}"
    )
    assert false_positive_rate <= 0.05, (
        f"False Positives rate {false_positive_rate:.1%} exceeds the 5% target. "
        f"The cache is returning false hits for adversarial queries.\n"
        f"First 10 leaks: {false_positive_failures[:10]}"
    )
