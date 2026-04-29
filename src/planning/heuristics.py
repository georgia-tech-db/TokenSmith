import re
from src.config import RAGConfig
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
  • Keyword queries (acronym-heavy, e.g. ACID / WAL / MVCC) → dominated by exact
    token matches, benefit strongly from BM25.
"""

# All-caps 2-4 char tokens. Matches ACID, WAL, MVCC, RDBMS, SQL, etc.
_ACRONYM_PATTERN = re.compile(r"\b[A-Z]{2,4}\b")

# Explanatory triggers: "why ...", "how does/do/is/are ...", "what causes ...",
# "explain ...". Anchored to the start of the lowercased query so that
# unrelated queries that merely contain the word "explain" in the middle
# don't get reclassified.
_EXPLANATORY_PATTERN = re.compile(r"^(why|how\s+(does|do|is|are)|what\s+causes|explain)\b")


class HeuristicQueryPlanner(QueryPlanner):
    @property
    def name(self) -> str:
        return "HeuristicBasedPlanner"

    def __init__(self, base_cfg: RAGConfig):
        super().__init__(base_cfg)
        self.base_cfg = deepcopy(base_cfg)

    def classify(self, query: str) -> str:
        # Acronym check runs on the original casing, before lowercasing.
        # It takes priority because queries like "what is ACID?" should be
        # routed to BM25-heavy retrieval instead of the generic definition
        # path.
        if _ACRONYM_PATTERN.search(query):
            return "keyword"
        q = query.lower()
        if any(x in q for x in ["compare", "comparison", "difference between", "differences between", "vs", "versus", "contrast"]):
            return "comparison"
        if any(x in q for x in ["what is", "define", "definition"]):
            return "definition"
        if any(x in q for x in ["how to", "steps", "procedure", "algorithm"]):
            return "procedural"
        if _EXPLANATORY_PATTERN.match(q) or "because" in q:
            return "explanatory"
        return "other"

    def plan(self, query: str) -> RAGConfig:
        kind = self.classify(query)
        cfg = deepcopy(self.base_cfg)

        if kind == "keyword":
            cfg.ranker_weights = {"faiss": 0.1, "bm25": 0.9}

        elif kind == "comparison":
            cfg.ranker_weights = {"faiss": 0.2, "bm25": 0.8}

        elif kind == "definition":
            cfg.ranker_weights = {"faiss": 0.3, "bm25": 0.7}

        elif kind == "explanatory":
            cfg.ranker_weights = {"faiss": 0.7, "bm25": 0.3}

        elif kind == "procedural":
            cfg.num_candidates = max(cfg.num_candidates, cfg.top_k * 5)
            cfg.ranker_weights = {"faiss": 0.6, "bm25": 0.4}

        else:
            print("Unknown query type. Defaulting to explanatory.")
            cfg.ranker_weights = {"faiss": 0.7, "bm25": 0.3}

        print(f"[PLANNER] HeuristicQueryPlanner: classified as {kind}, weights -> {cfg.ranker_weights}")
        self._log_decision(cfg)
        return cfg
