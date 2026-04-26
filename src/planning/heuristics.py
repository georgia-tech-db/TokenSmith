from copy import deepcopy

from src.config import RAGConfig
from src.planning.planner import QueryPlanner
from src.planning.rules import (
    QUERY_TYPE_DEFINITION,
    QUERY_TYPE_EXPLANATORY,
    QUERY_TYPE_FOLLOW_UP,
    QUERY_TYPE_MULTI_PART,
    QUERY_TYPE_OTHER,
    QUERY_TYPE_PROCEDURAL,
    classify_query,
)


class HeuristicQueryPlanner(QueryPlanner):
    """Legacy config-mutating planner that mirrors the shared routing vocabulary."""

    @property
    def name(self) -> str:
        return "HeuristicBasedPlanner"

    def __init__(self, base_cfg: RAGConfig):
        super().__init__(base_cfg)
        self.base_cfg = deepcopy(base_cfg)

    def classify(self, query: str) -> str:
        """Classify a query using the shared route policy, defaulting to explanatory."""
        decision = classify_query(query, has_history=True)
        return decision.query_type if decision.query_type != QUERY_TYPE_OTHER else QUERY_TYPE_EXPLANATORY

    def plan(self, query: str) -> RAGConfig:
        """Return a copied config with heuristic ranker weights for the query type."""
        kind = self.classify(query)
        cfg = deepcopy(self.base_cfg)

        if kind == QUERY_TYPE_DEFINITION:
            cfg.ranker_weights = {"faiss": 0.3, "bm25": 0.7, "index_keywords": 0.0}

        elif kind in {QUERY_TYPE_EXPLANATORY, QUERY_TYPE_FOLLOW_UP}:
            cfg.ranker_weights = {"faiss": 0.7, "bm25": 0.3, "index_keywords": 0.0}

        elif kind == QUERY_TYPE_PROCEDURAL:
            cfg.num_candidates = max(cfg.num_candidates, cfg.top_k * 5)
            cfg.ranker_weights = {"faiss": 0.6, "bm25": 0.4, "index_keywords": 0.0}

        elif kind == QUERY_TYPE_MULTI_PART:
            cfg.num_candidates = max(cfg.num_candidates, cfg.top_k * 6)
            cfg.ranker_weights = {"faiss": 0.55, "bm25": 0.35, "index_keywords": 0.10}

        self._log_decision(cfg, extra_info={"query_type": kind})
        return cfg
