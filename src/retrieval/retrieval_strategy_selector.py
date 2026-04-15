"""
Adaptive Retrieval Strategy Selector for TokenSmith.

Analyzes each incoming query and selects the most appropriate retrieval
configuration based on query features: length, intent signals (explanatory /
procedural / definition keywords), and structural cues (multi-part questions).

Pipeline placement:
    Query → RetrievalStrategySelector → (modified RAGConfig) → Retrievers → Ranker → Generator

Supported strategies
--------------------
BM25_ONLY   – Lexical retrieval (high BM25 weight).
              Suited for short, keyword-style queries.

DENSE_ONLY  – Semantic / dense retrieval (high FAISS weight).
              Suited for conceptual, explanatory, or definition queries.

HYBRID      – Balanced lexical + dense retrieval.
              Suited for multi-part queries or when no strong signal is present.

HYDE_DENSE  – HyDE-enhanced dense retrieval (generates a hypothetical answer
              before embedding the query).
              Suited for complex, analytical, or open-ended questions.
"""

from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Tuple

from src.config import RAGConfig


# ---------------------------------------------------------------------------
# Strategy enum
# ---------------------------------------------------------------------------

class RetrievalStrategy(Enum):
    """Available retrieval strategies."""
    BM25_ONLY  = "bm25_only"    # Lexical retrieval
    DENSE_ONLY = "dense_only"   # Semantic (FAISS) retrieval
    HYBRID     = "hybrid"       # Balanced lexical + dense
    HYDE_DENSE = "hyde_dense"   # HyDE-enhanced dense retrieval


# ---------------------------------------------------------------------------
# Intent keyword sets
# ---------------------------------------------------------------------------

_EXPLANATORY_TERMS: frozenset = frozenset([
    "why", "explain", "because", "reason", "purpose", "meaning",
    "describe", "clarify", "elaborate", "significance",
])

_PROCEDURAL_TERMS: frozenset = frozenset([
    "how to", "steps", "procedure", "algorithm", "process",
    "implement", "build", "create", "design", "walk me through",
])

_DEFINITION_TERMS: frozenset = frozenset([
    "what is", "what are", "define", "definition",
    "difference between", "compare", "contrast", "list",
])

_COMPLEX_TERMS: frozenset = frozenset([
    "analyze", "discuss", "elaborate", "evaluate", "assess",
    "examine", "critically", "tradeoff", "trade-off",
    "advantages and disadvantages", "pros and cons",
])


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class StrategyDecision:
    """Encapsulates the strategy choice and the ready-to-use modified config."""
    strategy: RetrievalStrategy
    reason: str
    modified_cfg: RAGConfig


# ---------------------------------------------------------------------------
# Selector
# ---------------------------------------------------------------------------

class RetrievalStrategySelector:
    """
    Rule-based adaptive retrieval strategy selector.

    Usage
    -----
    >>> selector = RetrievalStrategySelector(base_cfg)
    >>> decision = selector.select(query)
    >>> # decision.modified_cfg has updated ranker_weights and use_hyde
    >>> # decision.strategy / decision.reason explain the choice

    Classification priority (highest to lowest)
    -------------------------------------------
    1. Complex / analytical query  →  HYDE_DENSE
    2. Multi-part query            →  HYBRID
    3. Conceptual / procedural / definition query  →  DENSE_ONLY
    4. Short keyword query (≤ SHORT_QUERY_THRESHOLD words)  →  BM25_ONLY
    5. Default (medium-length, ambiguous)  →  HYBRID
    """

    # Thresholds
    SHORT_QUERY_THRESHOLD: int = 6   # ≤ this many words → BM25_ONLY candidate
    COMPLEX_QUERY_THRESHOLD: int = 15  # ≥ this many words → HYDE_DENSE candidate

    # Per-strategy weight presets  (index_keywords preserved from base cfg)
    _STRATEGY_PRESETS = {
        RetrievalStrategy.BM25_ONLY: {
            "ranker_weights": {"faiss": 0.2, "bm25": 0.8},
            "use_hyde": False,
        },
        RetrievalStrategy.DENSE_ONLY: {
            "ranker_weights": {"faiss": 0.8, "bm25": 0.2},
            "use_hyde": False,
        },
        RetrievalStrategy.HYBRID: {
            "ranker_weights": {"faiss": 0.5, "bm25": 0.5},
            "use_hyde": False,
        },
        RetrievalStrategy.HYDE_DENSE: {
            "ranker_weights": {"faiss": 0.85, "bm25": 0.15},
            "use_hyde": True,
        },
    }

    def __init__(self, base_cfg: RAGConfig) -> None:
        self._base_cfg = base_cfg

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select(self, query: str) -> StrategyDecision:
        """
        Analyze *query* and return a :class:`StrategyDecision` that contains
        the chosen strategy, a human-readable reason, and a ready-to-use
        modified :class:`RAGConfig`.
        """
        strategy, reason = self._classify(query)
        modified_cfg = self._build_config(strategy)
        return StrategyDecision(strategy=strategy, reason=reason, modified_cfg=modified_cfg)

    # ------------------------------------------------------------------
    # Feature extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _word_count(query: str) -> int:
        return len(query.split())

    @staticmethod
    def _is_multi_part(query: str) -> bool:
        """Return True if the query appears to contain multiple sub-questions."""
        q = query.strip()
        # Two or more question marks
        if q.count("?") >= 2:
            return True
        # Semicolons separating clauses
        if ";" in q:
            return True
        # "and" used twice (e.g. "explain X and Y and Z")
        if len(re.findall(r"\band\b", q, re.IGNORECASE)) >= 2:
            return True
        # Comma-separated questions starting with question words
        if re.search(
            r"\b(what|how|why|when|which)\b.+,\s*\b(what|how|why|when|which)\b",
            q, re.IGNORECASE
        ):
            return True
        return False

    @staticmethod
    def _contains_any(text: str, terms: frozenset) -> bool:
        t = text.lower()
        return any(term in t for term in terms)

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def _classify(self, query: str) -> Tuple[RetrievalStrategy, str]:
        """Return (strategy, reason_string) for the given query."""
        q = query.strip()
        wc = self._word_count(q)
        has_complex   = self._contains_any(q, _COMPLEX_TERMS)
        has_semantic  = (
            self._contains_any(q, _EXPLANATORY_TERMS)
            or self._contains_any(q, _PROCEDURAL_TERMS)
            or self._contains_any(q, _DEFINITION_TERMS)
        )

        # Priority 1: Complex / analytical → HyDE + Dense
        if wc >= self.COMPLEX_QUERY_THRESHOLD or has_complex:
            return (
                RetrievalStrategy.HYDE_DENSE,
                (
                    f"Complex/analytical query "
                    f"(words={wc}, complex_terms={has_complex})"
                ),
            )

        # Priority 2: Multi-part → Hybrid
        if self._is_multi_part(q):
            return (
                RetrievalStrategy.HYBRID,
                "Multi-part query detected (multiple sub-questions or clauses)",
            )

        # Priority 3: Conceptual / procedural / definition → Dense
        if has_semantic:
            return (
                RetrievalStrategy.DENSE_ONLY,
                "Conceptual/explanatory/definitional query (semantic retrieval preferred)",
            )

        # Priority 4: Short keyword → BM25
        if wc <= self.SHORT_QUERY_THRESHOLD:
            return (
                RetrievalStrategy.BM25_ONLY,
                f"Short keyword query (words={wc}, no strong semantic signal)",
            )

        # Default: medium-length ambiguous query → Hybrid
        return (
            RetrievalStrategy.HYBRID,
            f"Default hybrid strategy (words={wc}, no dominant signal)",
        )

    # ------------------------------------------------------------------
    # Config builder
    # ------------------------------------------------------------------

    def _build_config(self, strategy: RetrievalStrategy) -> RAGConfig:
        """
        Return a deep-copied RAGConfig with strategy-specific overrides.

        The ``index_keywords`` weight from the base config is preserved: the
        faiss/bm25 weights from the preset are scaled down proportionally so
        that the three weights still sum to 1.
        """
        cfg = deepcopy(self._base_cfg)
        preset = self._STRATEGY_PRESETS[strategy]

        # Build new weight dict, preserving index_keywords if enabled
        new_weights: dict = dict(preset["ranker_weights"])
        index_kw_w = self._base_cfg.ranker_weights.get("index_keywords", 0.0)
        if index_kw_w > 0:
            scale = 1.0 - index_kw_w
            new_weights = {k: v * scale for k, v in new_weights.items()}
            new_weights["index_keywords"] = index_kw_w

        cfg.ranker_weights = new_weights
        cfg.use_hyde = preset["use_hyde"]

        return cfg
