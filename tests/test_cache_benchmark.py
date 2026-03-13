import pytest
import numpy as np
import yaml
from pathlib import Path
from unittest.mock import MagicMock
from src.config import RAGConfig
from src.cache import (
    SEMANTIC_CACHE,
    semantic_cache_store,
    semantic_cache_lookup,
    compute_question_embedding,
    make_cache_config_key,
    normalize_question,
)


# -----------------------------
# Data loading
# -----------------------------

def load_benchmark_data():
    """Load benchmark questions from YAML."""
    yaml_path = Path(__file__).parent / "cache_benchmark.yaml"
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


BENCHMARK_DATA = load_benchmark_data()


# -----------------------------
# Fixtures
# -----------------------------

@pytest.fixture
def mock_config():
    """Create a mock RAGConfig for testing."""
    config = MagicMock(spec=RAGConfig)
    config.gen_model = "mock-model"
    config.embed_model = "models/Qwen3-Embedding-4B-Q5_K_M.gguf"
    config.top_k = 5
    config.system_prompt_mode = "baseline"
    config.ensemble_method = "rrf"
    config.ranker_weights = {"faiss": 0.5, "bm25": 0.5}
    config.use_hyde = False
    config.use_indexed_chunks = False
    config.disable_chunks = False
    config.use_golden_chunks = False
    config.semantic_cache_enabled = True
    return config


# -----------------------------
# Helpers
# -----------------------------

def print_separator():
    print("*" * 65)


def print_question_block(question_id, main_question, var_results, trick_results):
    """
    Print a formatted block for a single question with tick/X results
    for both variations and trick_variations.

    var_results:   list of (question_str, hit: bool)
    trick_results: list of (question_str, hit: bool)
    """
    print_separator()
    print(f"  [{question_id}] {main_question}")
    print()

    print("  Variations (hits are good ✅):")
    for q, hit in var_results:
        symbol = "✅" if hit else "❌"
        print(f"    {symbol}  {q}")

    print()
    print("  Trick Variations (hits are bad ⚠️):")
    for q, hit in trick_results:
        symbol = "⚠️ " if hit else "✅"
        print(f"    {symbol}  {q}")

    var_hits = sum(1 for _, h in var_results if h)
    trick_hits = sum(1 for _, h in trick_results if h)
    print()
    print(f"  Recall : {var_hits}/{len(var_results)} variations matched")
    print(f"  Leakage: {trick_hits}/{len(trick_results)} trick variations falsely matched")


# -----------------------------
# Test
# -----------------------------

def test_cache_benchmark_comprehensive(mock_config):
    """
    Benchmark the semantic cache against 30 questions, each with:
      - 5 genuine paraphrase variations  (hits expected)
      - 5 trick variations               (hits NOT expected)

    Scores:
      Recall rate  — fraction of genuine variations that got a cache hit.
                     Higher is better. Target >= 80%.
      Leakage rate — fraction of trick variations that falsely got a cache hit.
                     Lower is better. Target <= 10%.
    """
    # Clear any state from previous runs
    SEMANTIC_CACHE.clear()

    args = MagicMock()
    args.model_path = None
    args.system_prompt_mode = None
    args.index_prefix = "test_index"

    cache_key = make_cache_config_key(mock_config, args, None)
    embed_model_name = mock_config.embed_model

    total_var_hits = 0
    total_variations = 0
    total_trick_hits = 0
    total_tricks = 0

    recall_failures = []
    leakage_failures = []

    print(f"\n{'*' * 65}")
    print(f"  SEMANTIC CACHE BENCHMARK")
    print(f"  Embedding model : {embed_model_name}")
    print(f"  Questions       : {len(BENCHMARK_DATA)}")
    print(f"  Variations each : 5 genuine  +  5 trick")
    print(f"{'*' * 65}")

    for entry in BENCHMARK_DATA:
        question_id    = entry["id"]
        main_question  = entry["question"]
        variations     = entry.get("variations", [])
        trick_vars     = entry.get("trick_variations", [])

        # --- Seed the cache with the canonical question ---
        normalized_main = normalize_question(main_question)
        embedding_main  = compute_question_embedding(normalized_main, [], embed_model_name)
        assert embedding_main is not None, (
            f"Failed to compute embedding for {question_id}: '{main_question}'"
        )

        payload = {
            "answer":        f"Cached answer for {question_id}",
            "chunks_info":   [],
            "hyde_query":    None,
            "chunk_indices": [],
        }
        semantic_cache_store(cache_key, normalized_main, embedding_main, payload)

        # --- Test genuine variations ---
        var_results = []
        for var_q in variations:
            normalized_var = normalize_question(var_q)
            embedding_var  = compute_question_embedding(normalized_var, [], embed_model_name)
            hit = semantic_cache_lookup(cache_key, embedding_var, normalized_var) is not None
            var_results.append((var_q, hit))
            if not hit:
                recall_failures.append(f"[{question_id}] Missed variation : '{var_q}'")

        # --- Test trick variations ---
        trick_results = []
        for trick_q in trick_vars:
            normalized_trick = normalize_question(trick_q)
            embedding_trick  = compute_question_embedding(normalized_trick, [], embed_model_name)
            hit = semantic_cache_lookup(cache_key, embedding_trick, normalized_trick) is not None
            trick_results.append((trick_q, hit))
            if hit:
                leakage_failures.append(f"[{question_id}] False hit on trick: '{trick_q}'")

        # --- Accumulate totals ---
        total_var_hits   += sum(1 for _, h in var_results   if h)
        total_variations += len(var_results)
        total_trick_hits += sum(1 for _, h in trick_results if h)
        total_tricks     += len(trick_results)

        # --- Print per-question block ---
        print_question_block(question_id, main_question, var_results, trick_results)

    # --- Final summary ---
    recall_rate  = total_var_hits   / total_variations if total_variations > 0 else 0.0
    leakage_rate = total_trick_hits / total_tricks     if total_tricks     > 0 else 0.0

    print_separator()
    print()
    print(f"  {'FINAL BENCHMARK RESULTS':^61}")
    print()
    print(f"  {'Metric':<35} {'Score':>10}   {'Count'}")
    print(f"  {'-'*60}")
    print(f"  {'Recall Rate  (higher is better)':<35} {recall_rate:>9.1%}   {total_var_hits}/{total_variations}")
    print(f"  {'Leakage Rate (lower  is better)':<35} {leakage_rate:>9.1%}   {total_trick_hits}/{total_tricks}")
    print()
    print("  What these scores mean:")
    print()
    print("  Recall Rate — measures how often the cache correctly")
    print("  recognises a genuine paraphrase of a cached question.")
    print("  A high recall rate means users asking the same thing")
    print("  in different words will get a fast cached response.")
    print("  Target: >= 80%")
    print()
    print("  Leakage Rate — measures how often the cache is fooled")
    print("  into returning an answer for a semantically or")
    print("  syntactically similar but DIFFERENT question. A false")
    print("  hit here means a user gets the wrong cached answer.")
    print("  Lower is strictly better. Target: <= 10%")

    if recall_failures:
        print()
        print(f"  Recall misses ({len(recall_failures)} total, showing first 10):")
        for msg in recall_failures[:10]:
            print(f"    ❌ {msg}")

    if leakage_failures:
        print()
        print(f"  Leakage hits ({len(leakage_failures)} total, showing first 10):")
        for msg in leakage_failures[:10]:
            print(f"    ⚠️  {msg}")

    print()
    print_separator()

    # --- Assertions ---
    assert recall_rate >= 0.80, (
        f"Recall rate {recall_rate:.1%} is below the 80% target. "
        f"The cache is missing too many genuine paraphrases.\n"
        f"First 10 misses: {recall_failures[:10]}"
    )
    assert leakage_rate <= 0.10, (
        f"Leakage rate {leakage_rate:.1%} exceeds the 10% target. "
        f"The cache is returning false hits for trick questions.\n"
        f"First 10 leaks: {leakage_failures[:10]}"
    )