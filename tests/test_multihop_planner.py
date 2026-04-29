"""Unit tests for MultiHopQueryPlanner decomposition.

Uses monkeypatch to replace the LLM-backed decompose call with a stub so
these tests stay artifact-free and deterministic.
"""
from __future__ import annotations

import pytest

from src.config import RAGConfig
from src.planning.multihop import MultiHopQueryPlanner


@pytest.fixture()
def base_cfg() -> RAGConfig:
    return RAGConfig()


def test_multihop_truncates_to_cap(base_cfg, monkeypatch):
    planner = MultiHopQueryPlanner(base_cfg, max_subquestions=2)

    # Pretend the LLM returned 4 sub-questions; the planner must cap to 2.
    monkeypatch.setattr(
        "src.planning.multihop.decompose_complex_query",
        lambda query, model_path, **kwargs: [
            "what is FAISS",
            "what is BM25",
            "how does FAISS rank",
            "how does BM25 rank",
        ],
    )
    subs = planner._decompose("compare FAISS and BM25 retrieval performance")
    assert len(subs) <= 2, f"got {len(subs)} subquestions: {subs}"
    assert subs == ["what is FAISS", "what is BM25"]


def test_multihop_fallback_on_empty_llm_output(base_cfg, monkeypatch):
    planner = MultiHopQueryPlanner(base_cfg, max_subquestions=2)
    monkeypatch.setattr(
        "src.planning.multihop.decompose_complex_query",
        lambda query, model_path, **kwargs: [],
    )
    query = "compare FAISS and BM25"
    subs = planner._decompose(query)
    assert subs == [query], "must fall back to original query, never return zero subs"


def test_multihop_default_cap_is_three(base_cfg):
    assert MultiHopQueryPlanner(base_cfg).max_subquestions == 3


def test_multihop_dedupes_before_capping(base_cfg, monkeypatch):
    # Dedupe must happen before the cap, otherwise duplicates eat the budget.
    planner = MultiHopQueryPlanner(base_cfg, max_subquestions=2)
    monkeypatch.setattr(
        "src.planning.multihop.decompose_complex_query",
        lambda query, model_path, **kwargs: ["x", "X", "x", "y", "z"],
    )
    assert planner._decompose("some query") == ["x", "y"]
