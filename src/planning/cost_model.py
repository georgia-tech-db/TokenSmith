"""
Cost-Model Query Planner
------------------------
Routes each query to a sub-planner based on a per-category routing table.
The category is produced by an inner classifier (HeuristicQueryPlanner).

The routing table is empirically derived: for each category we use whichever
sub-planner had the higher answer-hit rate on the eval set. Categories not in
the table fall back to the configured default sub-planner.

This is the v1 cost model — a static lookup. The 125% goal replaces the
classifier with a learned model and the lookup with a confidence-scored
selection, but the planner contract stays the same.

Both `plan()` and `expand_queries()` are delegated to the chosen sub-planner
so multi-query expansion (e.g. multi-hop sub-questions) still happens on
routes that go to a composite planner.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from src.config import RAGConfig
from src.planning.heuristics import HeuristicQueryPlanner
from src.planning.planner import QueryPlanner


class CostModelPlanner(QueryPlanner):
    def __init__(
        self,
        base_cfg: RAGConfig,
        routing_table: Dict[str, QueryPlanner],
        default_planner: QueryPlanner,
        classifier: Optional[HeuristicQueryPlanner] = None,
    ):
        super().__init__(base_cfg)
        if not routing_table:
            raise ValueError("CostModelPlanner requires a non-empty routing_table.")
        self.routing_table = dict(routing_table)
        self.default_planner = default_planner
        self.classifier = classifier or HeuristicQueryPlanner(base_cfg)
        # Last-decision trace for observability — populated by plan()/expand_queries()
        self.last_decision: Dict[str, str] = {}

    @property
    def name(self) -> str:
        routes = ",".join(
            f"{cat}->{p.name}" for cat, p in sorted(self.routing_table.items())
        )
        return f"CostModel[{routes}|default={self.default_planner.name}]"

    def _route(self, query: str) -> tuple[str, QueryPlanner]:
        category = self.classifier.classify(query)
        chosen = self.routing_table.get(category, self.default_planner)
        self.last_decision = {
            "category": category,
            "chosen": chosen.name,
            "in_table": str(category in self.routing_table),
        }
        return category, chosen

    def plan(self, query: str) -> RAGConfig:
        category, chosen = self._route(query)
        print(
            f"[PLANNER] CostModelPlanner: category={category} -> {chosen.name}"
        )
        return chosen.plan(query)

    def expand_queries(self, query: str) -> List[str]:
        # Use cached decision if plan() was just called for the same query;
        # otherwise re-route. Re-routing is cheap (regex match) so it's fine
        # to do unconditionally — keeps this method side-effect-correct when
        # called before plan().
        _, chosen = self._route(query)
        return chosen.expand_queries(query)
