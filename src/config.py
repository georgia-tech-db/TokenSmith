from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Callable
import yaml

@dataclass
class QueryPlanConfig:
    # retrieval + ranking
    top_k: int = 5
    pool_size: int = 60
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    ensemble_method: str = "linear"   # "linear" | "rrf"
    rrf_k: int = 60
    ranker_weights: Dict[str, float] = field(
        default_factory=lambda: {"faiss": 0.6, "bm25": 0.25, "tf-idf": 0.15}
    )
    halo_mode: str = "none"
    seg_filter: Callable = None

    # generation
    max_gen_tokens: int = 400

    # # multi-index routing
    # default_index: str = "textbook_index"
    # routing_rules: List[Dict[str, str]] = field(default_factory=list)
    # # each rule: {"contains":"btree","then_index":"index_prefix"} or {"regex": "pattern", ...}

    # ---------- factory + validation ----------
    @staticmethod
    def from_yaml(path: str) -> "QueryPlanConfig":
        raw = yaml.safe_load(open(path))

        def pick(key, default=None):
            return raw.get(key, default)

        cfg = QueryPlanConfig(
            top_k          = pick("top_k", 5),
            pool_size      = pick("pool_size", 60),
            embed_model    = pick("embed_model", "sentence-transformers/all-MiniLM-L6-v2"),
            ensemble_method= pick("ensemble_method", "linear"),
            rrf_k          = pick("rrf_k", 60),
            ranker_weights = pick("ranker_weights", {"faiss":0.6,"bm25":0.25,"tf-idf":0.15}),
            max_gen_tokens = pick("max_gen_tokens", 400),
            halo_mode      = pick("halo_mode", "none"),
            # default_index  = pick("default_index", pick("index_prefix", "textbook_index")),
            # routing_rules  = pick("routing_rules", []),
        )
        cfg._validate()
        return cfg

    def _validate(self) -> None:
        assert self.top_k > 0, "top_k must be > 0"
        assert self.pool_size >= self.top_k, "pool_size must be >= top_k"
        assert self.ensemble_method.lower() in {"linear","weighted","rrf"}
        if self.ensemble_method.lower() in {"linear","weighted"}:
            s = sum(self.ranker_weights.values()) or 1.0
            self.ranker_weights = {k: v/s for k, v in self.ranker_weights.items()}

    # def choose_index(self, query: str) -> str:
    #     ql = query.lower()
    #     for rule in self.routing_rules:
    #         if "contains" in rule and rule["contains"].lower() in ql:
    #             return rule.get("then_index", self.default_index)
    #         if "regex" in rule and re.search(rule["regex"], query, flags=re.IGNORECASE):
    #             return rule.get("then_index", self.default_index)
    #     return self.default_index
