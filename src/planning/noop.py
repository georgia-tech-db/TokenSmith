"""
No-op planner: returns the base cfg unchanged and never expands the query.
Useful as a baseline reference and as a building block for higher-level
planners (e.g. CostModelPlanner) that want to short-circuit to "do nothing"
on certain query categories.
"""
from __future__ import annotations

from copy import deepcopy

from src.config import RAGConfig
from src.planning.planner import QueryPlanner


class NoOpPlanner(QueryPlanner):
    @property
    def name(self) -> str:
        return "NoOpPlanner"

    def plan(self, query: str) -> RAGConfig:
        return deepcopy(self.base_cfg)
