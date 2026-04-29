"""
Multi-hop Query Planner
-----------------------
Detects queries that ask about multiple concepts at once ("compare X and Y",
"what is A and why does B happen") and decomposes them into sub-questions.
Each sub-question is retrieved for independently; the callers in the pipeline
merge and deduplicate the resulting chunks before ranking.

The planner still participates in the `QueryPlanner` cfg-mutation contract:
for detected multi-hop queries it widens the candidate pool so that merging
has enough headroom. The actual sub-question expansion is exposed via the
`expand_queries` hook on the base class.
"""
from __future__ import annotations

import re
from copy import deepcopy
from typing import List

from src.config import RAGConfig
from src.planning.planner import QueryPlanner
from src.query_enhancement import decompose_complex_query


_MULTIHOP_PATTERNS = [
    re.compile(r"\bcompare\b", re.IGNORECASE),
    re.compile(r"\bcontrast\b", re.IGNORECASE),
    re.compile(r"\bdifference(s)?\s+between\b", re.IGNORECASE),
    re.compile(r"\bvs\.?\b", re.IGNORECASE),
    re.compile(r"\bversus\b", re.IGNORECASE),
    re.compile(r"\bboth\b", re.IGNORECASE),
    re.compile(r"\beach\s+of\b", re.IGNORECASE),
    re.compile(r"\band\s+(also|how|why|what|when|where)\b", re.IGNORECASE),
]


class MultiHopQueryPlanner(QueryPlanner):
    @property
    def name(self) -> str:
        return "MultiHopPlanner"

    def __init__(self, base_cfg: RAGConfig, max_subquestions: int = 3):
        super().__init__(base_cfg)
        # RRF dilutes over too many retrieval sets — cap sub-question count
        # so comparison queries don't get starved.
        self.max_subquestions = max(1, int(max_subquestions))
        # Cache decompositions so plan() and expand_queries() don't call the
        # LLM twice for the same question within a single turn.
        self._decomposition_cache: dict[str, List[str]] = {}

    def _looks_multihop(self, query: str) -> bool:
        if any(p.search(query) for p in _MULTIHOP_PATTERNS):
            return True
        # Two or more question marks => multiple direct questions joined.
        if query.count("?") >= 2:
            return True
        return False

    def _decompose(self, query: str) -> List[str]:
        if query in self._decomposition_cache:
            return self._decomposition_cache[query]

        try:
            raw = decompose_complex_query(query, self.base_cfg.gen_model)
        except Exception:
            raw = [query]

        seen: set[str] = set()
        subs: List[str] = []
        for item in raw:
            if not isinstance(item, str):
                continue
            cleaned = item.strip()
            if not cleaned:
                continue
            key = cleaned.lower()
            if key in seen:
                continue
            seen.add(key)
            subs.append(cleaned)

        # Truncate post-hoc: the LLM frequently ignores the "at most N"
        # instruction, so we enforce the cap on our side.
        subs = subs[: self.max_subquestions]

        # If the LLM collapsed to nothing useful, fall back to the original.
        if not subs or (len(subs) == 1 and subs[0].lower() == query.lower()):
            subs = [query]

        self._decomposition_cache[query] = subs
        return subs

    def plan(self, query: str) -> RAGConfig:
        cfg = deepcopy(self.base_cfg)

        is_multihop = self._looks_multihop(query)
        if is_multihop:
            subs = self._decompose(query)
            if len(subs) > 1:
                # Widen the per-retriever pool so that each sub-question
                # contributes candidates without starving the others.
                cfg.num_candidates = max(
                    cfg.num_candidates,
                    cfg.top_k * max(4, len(subs) * 2),
                )
        else:
            subs = [query]

        print(f"[PLANNER] MultiHopQueryPlanner: multihop={is_multihop}, sub_questions={subs}")
        self._log_decision(cfg)
        return cfg

    def expand_queries(self, query: str) -> List[str]:
        if not self._looks_multihop(query):
            return [query]
        return self._decompose(query)
