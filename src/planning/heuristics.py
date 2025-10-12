from src.config import QueryPlanConfig
from copy import deepcopy

from src.planning.planner import QueryPlanner

"""
Heuristic Query Planner
-----------------------
TODO: verify below assertions with data
- Different query types have different needs:
  • Definition queries → usually short answers, need fine-grained chunks (small tokens), 
    benefit from keyword match (BM25).
  • Explanatory queries → broader answers, need larger spans (sections), 
    benefit from semantic similarity (FAISS).
  • Procedural queries (how-to, steps) → benefit from wider candidate pools and tag overlap, 
    since relevant steps may be scattered.
"""
class HeuristicQueryPlanner(QueryPlanner):
    @property
    def name(self) -> str:
        return "HeuristicBasedPlanner"

    def __init__(self, base_cfg: QueryPlanConfig):
        super().__init__(base_cfg)
        self.base_cfg = deepcopy(base_cfg)

    def classify(self, query: str) -> str:
        q = query.lower()
        if any(x in q for x in ["what is", "define", "definition"]):
            return "definition"
        if any(x in q for x in ["why", "explain", "because"]):
            return "explanatory"
        if any(x in q for x in ["how to", "steps", "procedure", "algorithm"]):
            return "procedural"
        return "other"

    def plan(self, query: str) -> QueryPlanConfig:
        kind = self.classify(query)
        cfg = deepcopy(self.base_cfg)

        if kind == "definition":
            cfg.ranker_weights = {"faiss": 0.3, "bm25": 0.7}

        elif kind == "explanatory":
            cfg.ranker_weights = {"faiss": 0.7, "bm25": 0.3}

        elif kind == "procedural":
            cfg.pool_size = max(cfg.pool_size, cfg.top_k * 5)
            cfg.ranker_weights = {"faiss": 0.6, "bm25": 0.4}

        else:
            print("Unknown query type. Defaulting to explanatory.")
            cfg.ranker_weights = {"faiss": 0.7, "bm25": 0.3}

        self._log_decision(cfg)
        return cfg
