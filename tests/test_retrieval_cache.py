import pytest

from src.retrieval_cache import (
    RetrievalCacheRecord,
    RetrievalContextCache,
    build_cache_key,
    build_index_fingerprint,
    build_retrieval_signature,
    normalize_query,
)


@pytest.mark.unit
def test_normalize_query():
    """Ensures query normalization collapses spaces/case for stable cache keys."""
    assert normalize_query("  What   Is ACID? ") == "what is acid?"


@pytest.mark.unit
def test_cache_roundtrip(tmp_path):
    """Verifies a stored retrieval cache record can be loaded unchanged."""
    db_path = tmp_path / "retrieval_cache.db"
    cache = RetrievalContextCache(db_path=str(db_path), max_entries=10)

    key = "abc123"
    sig = {"top_k": 5}
    fp = "indexfp"
    record = RetrievalCacheRecord(
        topk_idxs=[1, 2, 3],
        ranked_chunks=["chunk-a", "chunk-b"],
        ordered_scores=[0.9, 0.8, 0.7],
    )
    cache.set(
        cache_key=key,
        normalized_query="what is acid",
        retrieval_signature=sig,
        index_fingerprint=fp,
        record=record,
    )

    loaded = cache.get(key)
    assert loaded is not None
    assert loaded.topk_idxs == [1, 2, 3]
    assert loaded.ranked_chunks == ["chunk-a", "chunk-b"]
    assert loaded.ordered_scores == [0.9, 0.8, 0.7]


@pytest.mark.unit
def test_cache_key_changes_with_signature_and_fingerprint():
    """Ensures cache keys change when retrieval settings or corpus fingerprint changes."""
    q = normalize_query("What is ACID")
    sig1 = {"top_k": 5, "ensemble_method": "rrf"}
    sig2 = {"top_k": 10, "ensemble_method": "rrf"}
    k1 = build_cache_key(q, sig1, "fp1")
    k2 = build_cache_key(q, sig2, "fp1")
    k3 = build_cache_key(q, sig1, "fp2")
    assert k1 != k2
    assert k1 != k3


@pytest.mark.unit
def test_index_fingerprint_changes_with_chunks():
    """Ensures corpus fingerprint detects chunk-content changes."""
    fp1 = build_index_fingerprint(["a", "b"], ["s1", "s2"], [{"page_numbers": [1]}, {"page_numbers": [2]}])
    fp2 = build_index_fingerprint(["a", "changed"], ["s1", "s2"], [{"page_numbers": [1]}, {"page_numbers": [2]}])
    assert fp1 != fp2


@pytest.mark.unit
def test_build_retrieval_signature_from_cfg():
    """Ensures retrieval signature includes expected config knobs for invalidation."""
    from src.config import RAGConfig

    cfg = RAGConfig(
        top_k=7,
        num_candidates=20,
        ensemble_method="rrf",
        ranker_weights={"faiss": 1.0, "bm25": 0.0, "index_keywords": 0.0},
    )
    sig = build_retrieval_signature(cfg)
    assert sig["top_k"] == 7
    assert sig["num_candidates"] == 20
    assert sig["ensemble_method"] == "rrf"


@pytest.mark.unit
def test_context_cache_used_only_when_enabled(tmp_path):
    """
    Validates cache-path behavior in `get_answer`:
    - With cache enabled: second identical query should skip retrievers (cache hit path).
    - With cache disabled: retrievers should be called again on the second query.
    """
    import argparse
    from unittest.mock import MagicMock, patch

    from src.config import RAGConfig
    from src.main import get_answer
    from src.ranking.ranker import EnsembleRanker

    class CountingRetriever:
        def __init__(self, name, scores):
            self.name = name
            self.scores = scores
            self.calls = 0

        def get_scores(self, query, pool_size, chunks):
            self.calls += 1
            return self.scores

    chunks = [
        "Chunk 0: ACID means atomicity consistency isolation durability.",
        "Chunk 1: Transactions support concurrent execution.",
        "Chunk 2: Recovery uses logs.",
    ]
    sources = ["doc", "doc", "doc"]
    meta = [{"page_numbers": [1]}, {"page_numbers": [2]}, {"page_numbers": [3]}]

    faiss = CountingRetriever("faiss", {0: 0.9, 1: 0.5, 2: 0.2})
    bm25 = CountingRetriever("bm25", {0: 0.8, 1: 0.6, 2: 0.1})
    ranker = EnsembleRanker("rrf", {"faiss": 0.5, "bm25": 0.5}, rrf_k=60)
    artifacts = {
        "chunks": chunks,
        "sources": sources,
        "retrievers": [faiss, bm25],
        "ranker": ranker,
        "meta": meta,
    }

    args = argparse.Namespace(
        index_prefix="textbook_index",
        model_path="dummy.gguf",
        system_prompt_mode="baseline",
        double_prompt=False,
    )
    logger = MagicMock()
    question = "What are the ACID properties of transactions?"

    # Keep generation deterministic and cheap for the test.
    def fake_stream():
        yield "stub answer"

    cache_db = str(tmp_path / "context_cache.db")
    with patch("src.main.answer", side_effect=lambda *a, **k: fake_stream()):
        cfg_on = RAGConfig(
            top_k=2,
            num_candidates=3,
            ensemble_method="rrf",
            ranker_weights={"faiss": 0.5, "bm25": 0.5, "index_keywords": 0.0},
            rerank_mode="none",
            enable_context_chunk_cache=True,
            retrieval_cache_path=cache_db,
            retrieval_cache_max_entries=50,
        )
        get_answer(question, cfg_on, args, logger, None, artifacts=artifacts, is_test_mode=True)
        get_answer(question, cfg_on, args, logger, None, artifacts=artifacts, is_test_mode=True)
        calls_with_cache = faiss.calls + bm25.calls

        # Reset call counters for cache-disabled check.
        faiss.calls = 0
        bm25.calls = 0

        cfg_off = RAGConfig(
            top_k=2,
            num_candidates=3,
            ensemble_method="rrf",
            ranker_weights={"faiss": 0.5, "bm25": 0.5, "index_keywords": 0.0},
            rerank_mode="none",
            enable_context_chunk_cache=False,
            retrieval_cache_path=cache_db,
            retrieval_cache_max_entries=50,
        )
        get_answer(question, cfg_off, args, logger, None, artifacts=artifacts, is_test_mode=True)
        get_answer(question, cfg_off, args, logger, None, artifacts=artifacts, is_test_mode=True)
        calls_without_cache = faiss.calls + bm25.calls

    # Two retrievers:
    # - cache ON: first query => 2 calls, second query => 0 calls (cache hit) => total 2
    # - cache OFF: first query => 2 calls, second query => 2 calls => total 4
    assert calls_with_cache == 2
    assert calls_without_cache == 4
