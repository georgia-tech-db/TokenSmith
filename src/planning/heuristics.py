from src.config import QueryPlanConfig
from src.chunking_constants import CHUNK_MODE_TOKENS, CHUNK_MODE_SECTIONS, CHUNK_MODE_SLIDING_TOKENS
from src.chunking import CharChunkConfig, LLMChunkConfig, PropositionalChunkConfig, TokenChunkConfig, SlidingTokenConfig, SectionChunkConfig
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
            cfg.chunk_mode = CHUNK_MODE_TOKENS
            cfg.chunk_config = SectionChunkConfig()
            cfg.ranker_weights = {"faiss": 0.3, "bm25": 0.6, "tf-idf": 0.1}

        elif kind == "explanatory":
            cfg.chunk_mode = CHUNK_MODE_SECTIONS
            cfg.chunk_config = SectionChunkConfig()
            cfg.ranker_weights = {"faiss": 0.7, "bm25": 0.2, "tf-idf": 0.1}

        elif kind == "procedural":
            cfg.chunk_mode = CHUNK_MODE_SLIDING_TOKENS
            cfg.chunk_config = SlidingTokenConfig(
                max_tokens=400,
                overlap_tokens=100,
                tokenizer_name=cfg.embed_model,
            )
            cfg.pool_size = max(cfg.pool_size, cfg.top_k * 5)
            cfg.ranker_weights = {"faiss": 0.5, "bm25": 0.2, "tf-idf": 0.3}

        else:
            print("Unknown query type. Defaulting to explanatory.")
            cfg.chunk_mode = CHUNK_MODE_SECTIONS
            cfg.chunk_config = SectionChunkConfig()
            cfg.ranker_weights = {"faiss": 0.7, "bm25": 0.2, "tf-idf": 0.1}

        self._log_decision(cfg)
        return cfg
