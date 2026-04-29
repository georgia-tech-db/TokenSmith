"""
Cost-Model Query Planner
------------------------
Routes each query to a sub-planner based on a per-category routing table.
The category is produced by an inner classifier (regex or learned).

The routing table is empirically derived: for each category we use whichever
sub-planner had the higher answer-hit rate on the eval set. Categories not in
the table fall back to the configured default sub-planner.

The classifier just needs to expose `classify_query(query) -> Classification`.
HeuristicQueryPlanner provides this via an adapter; PrototypeClassifier
implements it natively. This decouples routing from classifier shape so the
125% learned classifier swaps in without other code changes.

Both `plan()` and `expand_queries()` are delegated to the chosen sub-planner
so multi-query expansion (e.g. multi-hop sub-questions) still happens on
routes that go to a composite planner.

Confidence fallback (Phase 4 of 125% goal):
If the classifier's confidence on the top category is below
`confidence_threshold`, route to `fallback_planner` (defaults to NoOp via the
default_planner) regardless of the routing table. This trades incremental
gain for a guarantee that uncertain queries are never sent to a planner that
might hurt them — i.e., the cost model becomes monotonically non-worse than
baseline by construction.
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
        classifier=None,
        confidence_threshold: float = 0.0,
        fallback_planner: Optional[QueryPlanner] = None,
    ):
        super().__init__(base_cfg)
        if not routing_table:
            raise ValueError("CostModelPlanner requires a non-empty routing_table.")
        self.routing_table = dict(routing_table)
        self.default_planner = default_planner
        self.classifier = classifier if classifier is not None else HeuristicQueryPlanner(base_cfg)
        self.confidence_threshold = float(confidence_threshold)
        # If no explicit fallback planner is passed, fall back to default.
        # In typical wiring `default_planner` is the composite optimizer; for
        # safe-mode fallback the caller should pass NoOp explicitly.
        self.fallback_planner = fallback_planner if fallback_planner is not None else default_planner
        # Last-decision trace for observability
        self.last_decision: Dict[str, str] = {}

    @property
    def name(self) -> str:
        routes = ",".join(
            f"{cat}->{p.name}" for cat, p in sorted(self.routing_table.items())
        )
        suffix = ""
        if self.confidence_threshold > 0:
            suffix = f"|cf={self.confidence_threshold:.2f}->{self.fallback_planner.name}"
        return f"CostModel[{routes}|default={self.default_planner.name}{suffix}]"

    def _classify(self, query: str):
        """
        Adapter: support either classifier shape.
        - Native protocol: classifier has `classify_query` returning Classification
        - Legacy: classifier has `classify` returning a string (treated as confidence=1.0)
        """
        if hasattr(self.classifier, "classify_query"):
            return self.classifier.classify_query(query)
        # Legacy fallback — wrap the string return so the rest of the planner
        # only deals with the Classification shape.
        from src.planning.learned_classifier import Classification
        cat = self.classifier.classify(query)
        return Classification(category=cat, confidence=1.0, all_scores={cat: 1.0})

    def _route(self, query: str) -> tuple[str, QueryPlanner]:
        result = self._classify(query)
        category = result.category
        confidence = float(result.confidence)

        if self.confidence_threshold > 0 and confidence < self.confidence_threshold:
            chosen = self.fallback_planner
            fallback = True
            in_table = False
        else:
            chosen = self.routing_table.get(category, self.default_planner)
            fallback = False
            in_table = category in self.routing_table

        self.last_decision = {
            "category": category,
            "confidence": f"{confidence:.4f}",
            "chosen": chosen.name,
            "in_table": str(in_table),
            "fallback": str(fallback),
        }
        return category, chosen

    def plan(self, query: str) -> RAGConfig:
        category, chosen = self._route(query)
        decision = self.last_decision
        print(
            f"[PLANNER] CostModelPlanner: category={category} "
            f"conf={decision['confidence']} fallback={decision['fallback']} "
            f"-> {chosen.name}"
        )
        return chosen.plan(query)

    def expand_queries(self, query: str) -> List[str]:
        # Re-routing is cheap (one classifier call) so it's fine to do
        # unconditionally — keeps this method side-effect-correct when
        # called before plan().
        _, chosen = self._route(query)
        return chosen.expand_queries(query)
