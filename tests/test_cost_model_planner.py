"""Unit tests for CostModelPlanner routing.

Stubs the sub-planners and the classifier so the test exercises only the
routing logic, not retrieval or LLM calls.
"""
from __future__ import annotations

from typing import List

import pytest

from src.config import RAGConfig
from src.planning.cost_model import CostModelPlanner
from src.planning.heuristics import HeuristicQueryPlanner
from src.planning.planner import QueryPlanner


class _StubPlanner(QueryPlanner):
    """Records calls; returns the cfg with a tag injected so tests can detect
    which sub-planner produced the result."""

    def __init__(self, base_cfg: RAGConfig, tag: str, expansion: List[str] | None = None):
        super().__init__(base_cfg)
        self.tag = tag
        self._expansion = expansion or [f"{tag}_query"]
        self.plan_calls: List[str] = []
        self.expand_calls: List[str] = []

    @property
    def name(self) -> str:
        return f"Stub[{self.tag}]"

    def plan(self, query: str) -> RAGConfig:
        self.plan_calls.append(query)
        cfg = RAGConfig()
        # Use ranker_weights as a side channel to identify the planner — the
        # field already exists on RAGConfig so we don't need to extend it.
        cfg.ranker_weights = {self.tag: 1.0}
        return cfg

    def expand_queries(self, query: str) -> List[str]:
        self.expand_calls.append(query)
        return self._expansion


@pytest.fixture()
def base_cfg() -> RAGConfig:
    return RAGConfig()


@pytest.fixture()
def stubs(base_cfg):
    return {
        "composite": _StubPlanner(base_cfg, "composite", expansion=["sub1", "sub2"]),
        "noop": _StubPlanner(base_cfg, "noop"),
    }


@pytest.fixture()
def planner(base_cfg, stubs):
    routing = {
        "keyword":     stubs["composite"],
        "definition":  stubs["composite"],
        "procedural":  stubs["composite"],
        "other":       stubs["composite"],
        "comparison":  stubs["noop"],
        "explanatory": stubs["noop"],
    }
    return CostModelPlanner(
        base_cfg,
        routing_table=routing,
        default_planner=stubs["composite"],
        classifier=HeuristicQueryPlanner(base_cfg),
    )


def test_comparison_routes_to_noop(planner, stubs):
    cfg = planner.plan("Compare clustered and non-clustered indexes")
    assert "noop" in cfg.ranker_weights
    assert stubs["noop"].plan_calls == ["Compare clustered and non-clustered indexes"]
    assert stubs["composite"].plan_calls == []


def test_explanatory_routes_to_noop(planner, stubs):
    # Avoid acronyms — the heuristic's acronym check fires before the
    # explanatory check, so a query like "Why is BCNF..." routes as keyword.
    query = "Why does write-ahead logging avoid losing committed transactions"
    cfg = planner.plan(query)
    assert "noop" in cfg.ranker_weights
    assert stubs["noop"].plan_calls == [query]


def test_keyword_routes_to_composite(planner, stubs):
    cfg = planner.plan("What is ACID?")
    assert "composite" in cfg.ranker_weights
    assert stubs["composite"].plan_calls == ["What is ACID?"]
    assert stubs["noop"].plan_calls == []


def test_definition_routes_to_composite(planner, stubs):
    cfg = planner.plan("What is a foreign key")
    assert "composite" in cfg.ranker_weights


def test_procedural_routes_to_composite(planner, stubs):
    cfg = planner.plan("How to perform two-phase commit")
    assert "composite" in cfg.ranker_weights


def test_unknown_category_uses_default(base_cfg, stubs):
    """If the routing table doesn't list a category, fall back to default."""
    fallback = _StubPlanner(base_cfg, "fallback")
    planner = CostModelPlanner(
        base_cfg,
        routing_table={"keyword": stubs["composite"]},  # only one entry
        default_planner=fallback,
        classifier=HeuristicQueryPlanner(base_cfg),
    )
    cfg = planner.plan("just some random sentence")  # → "other"
    assert "fallback" in cfg.ranker_weights
    assert fallback.plan_calls == ["just some random sentence"]


def test_expand_queries_delegated_to_chosen_planner(planner, stubs):
    # Multi-hop-style expansion should pass through when routed to composite.
    expanded = planner.expand_queries("What is ACID?")
    assert expanded == ["sub1", "sub2"]
    assert stubs["composite"].expand_calls == ["What is ACID?"]
    assert stubs["noop"].expand_calls == []


def test_expand_queries_uses_noop_for_routed_categories(planner, stubs):
    # Comparison routes to noop; noop returns its own (single-element) expansion.
    expanded = planner.expand_queries("Compare X and Y")
    assert expanded == ["noop_query"]
    assert stubs["noop"].expand_calls == ["Compare X and Y"]


def test_last_decision_records_route(planner):
    planner.plan("What is ACID?")
    assert planner.last_decision["category"] == "keyword"
    assert "composite" in planner.last_decision["chosen"]
    assert planner.last_decision["in_table"] == "True"


def test_empty_routing_table_rejected(base_cfg, stubs):
    with pytest.raises(ValueError):
        CostModelPlanner(
            base_cfg,
            routing_table={},
            default_planner=stubs["composite"],
        )


# ---------------------------------------------------------------------------
# Phase 4: confidence-based fallback
# ---------------------------------------------------------------------------


class _StubClassifier:
    """Returns a Classification with a configurable category + confidence."""

    def __init__(self, category: str, confidence: float):
        self.category = category
        self.confidence = confidence
        self.calls: List[str] = []

    def classify_query(self, query: str):
        from src.planning.learned_classifier import Classification
        self.calls.append(query)
        return Classification(
            category=self.category,
            confidence=self.confidence,
            all_scores={self.category: self.confidence},
        )


def _make_planner(base_cfg, stubs, classifier, *, threshold, fallback=None):
    routing = {
        "keyword":     stubs["composite"],
        "comparison":  stubs["noop"],
    }
    return CostModelPlanner(
        base_cfg,
        routing_table=routing,
        default_planner=stubs["composite"],
        classifier=classifier,
        confidence_threshold=threshold,
        fallback_planner=fallback,
    )


def test_high_confidence_uses_routing_table(base_cfg, stubs):
    clf = _StubClassifier("keyword", confidence=0.95)
    fallback = _StubPlanner(base_cfg, "fallback")
    planner = _make_planner(base_cfg, stubs, clf, threshold=0.5, fallback=fallback)

    cfg = planner.plan("query")
    assert "composite" in cfg.ranker_weights
    assert stubs["composite"].plan_calls == ["query"]
    assert fallback.plan_calls == []


def test_low_confidence_routes_to_fallback(base_cfg, stubs):
    clf = _StubClassifier("keyword", confidence=0.2)
    fallback = _StubPlanner(base_cfg, "fallback")
    planner = _make_planner(base_cfg, stubs, clf, threshold=0.5, fallback=fallback)

    cfg = planner.plan("query")
    assert "fallback" in cfg.ranker_weights
    assert fallback.plan_calls == ["query"]
    assert stubs["composite"].plan_calls == []
    assert stubs["noop"].plan_calls == []


def test_confidence_threshold_zero_disables_fallback(base_cfg, stubs):
    """Default threshold=0 means fallback never triggers — backward compat."""
    clf = _StubClassifier("comparison", confidence=0.01)  # very low
    planner = _make_planner(base_cfg, stubs, clf, threshold=0.0)

    cfg = planner.plan("query")
    # Should route via table (comparison -> noop), NOT fallback
    assert "noop" in cfg.ranker_weights
    assert stubs["noop"].plan_calls == ["query"]


def test_last_decision_includes_confidence_and_fallback(base_cfg, stubs):
    clf = _StubClassifier("keyword", confidence=0.3)
    fallback = _StubPlanner(base_cfg, "fallback")
    planner = _make_planner(base_cfg, stubs, clf, threshold=0.5, fallback=fallback)

    planner.plan("query")
    assert planner.last_decision["category"] == "keyword"
    assert planner.last_decision["confidence"] == "0.3000"
    assert planner.last_decision["fallback"] == "True"
    assert "fallback" in planner.last_decision["chosen"]


def test_fallback_planner_defaults_to_default_planner(base_cfg, stubs):
    """If no fallback_planner is given, low confidence routes to default."""
    clf = _StubClassifier("keyword", confidence=0.1)
    planner = _make_planner(base_cfg, stubs, clf, threshold=0.5)  # no fallback arg

    cfg = planner.plan("query")
    # default_planner is stubs["composite"] in _make_planner
    assert "composite" in cfg.ranker_weights


def test_legacy_classifier_still_works(base_cfg, stubs):
    """Plain HeuristicQueryPlanner (no classify_query) should still work
    via the duck-typed adapter in _classify."""

    class LegacyClassifier:
        def classify(self, query: str) -> str:
            return "keyword"

    planner = CostModelPlanner(
        base_cfg,
        routing_table={"keyword": stubs["composite"]},
        default_planner=stubs["noop"],
        classifier=LegacyClassifier(),
    )
    cfg = planner.plan("query")
    assert "composite" in cfg.ranker_weights
