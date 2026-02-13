from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict
from copy import deepcopy

from src.config import RAGConfig
from src.instrumentation.logging import get_logger


class QueryPlanner(ABC):
    """
    Abstract base for query planners.
    """

    def __init__(self, base_cfg: RAGConfig):
        self.base_cfg = deepcopy(base_cfg)

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name of the planner (for logging)."""

    @abstractmethod
    def plan(self, query: str) -> RAGConfig:
        """
        Subclasses must override this to return an updated RAGConfig.
        """

    # ---- helper for subclasses ----
    def _log_decision(self, new_cfg: RAGConfig) -> None:
        base_dict = self.base_cfg.to_dict()
        new_dict = new_cfg.to_dict()
        # TO DO - fix this
        # get_logger().log_planner(self.name, base_dict, new_dict)
