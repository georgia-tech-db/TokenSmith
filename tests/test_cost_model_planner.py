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
