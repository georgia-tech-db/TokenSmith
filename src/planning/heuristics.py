import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import QueryPlanConfig
from chunking import CharChunkConfig, TokenChunkConfig, SlidingTokenConfig, SectionChunkConfig
from copy import deepcopy
import re

from planning.planner import QueryPlanner

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
        # Check for location queries first
        if any(x in q for x in ["where is", "where can", "where do", "where does", "in which", "what section", "what chapter"]):
            return "location"
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

        # --- extract optional location hints ---
        # chapter: "chapter 19" or "ch. 19"
        # section: "section 19.3" or "§19.3"
        ch_match = re.search(r"\b(?:chapter|ch\.)\s*(\d{1,3})\b", query, flags=re.IGNORECASE)
        sec_match = re.search(r"\b(?:section|sec\.|§)\s*(\d{1,3}(?:\.\d{1,3})+)\b", query, flags=re.IGNORECASE)
        location_hint = None
        if ch_match or sec_match:
            location_hint = {
                "chapter": int(ch_match.group(1)) if ch_match else None,
                "section": sec_match.group(1) if sec_match else None,
                "raw": query,
            }
            cfg.location_hint = location_hint

        if kind == "location":
            cfg.chunk_mode = "sections"
            cfg.chunk_config = SectionChunkConfig()
            cfg.ranker_weights = {"faiss": 0.6, "bm25": 0.2, "tf-idf": 0.1, "location": 0.1}

        elif kind == "definition":
            cfg.chunk_mode = "tokens"
            cfg.chunk_config = TokenChunkConfig(max_tokens=200)
            cfg.ranker_weights = {"faiss": 0.3, "bm25": 0.6, "tf-idf": 0.1, "location": 0.0}

        elif kind == "explanatory":
            cfg.chunk_mode = "sections"
            cfg.chunk_config = SectionChunkConfig()
            cfg.ranker_weights = {"faiss": 0.7, "bm25": 0.2, "tf-idf": 0.1, "location": 0.0}

        elif kind == "procedural":
            cfg.chunk_mode = "sliding-tokens"
            cfg.chunk_config = SlidingTokenConfig(
                max_tokens=400,
                overlap_tokens=100,
                tokenizer_name=cfg.embed_model,
            )
            cfg.pool_size = max(cfg.pool_size, cfg.top_k * 5)
            cfg.ranker_weights = {"faiss": 0.5, "bm25": 0.2, "tf-idf": 0.3, "location": 0.0}

        else:
            print("Unknown query type. Defaulting to explanatory.")
            cfg.chunk_mode = "sections"
            cfg.chunk_config = SectionChunkConfig()
            cfg.ranker_weights = {"faiss": 0.7, "bm25": 0.2, "tf-idf": 0.1, "location": 0.0}

        # If location hints are present, boost location ranker weight
        if cfg.location_hint:
            # Reduce other weights proportionally to make room for location
            total_other = sum(v for k, v in cfg.ranker_weights.items() if k != "location")
            if total_other > 0:
                scale_factor = 0.9  # Reserve 10% for location (more conservative)
                for k in cfg.ranker_weights:
                    if k != "location":
                        cfg.ranker_weights[k] *= scale_factor
                cfg.ranker_weights["location"] = 0.1

        self._log_decision(cfg)
        return cfg
