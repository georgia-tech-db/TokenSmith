"""
query_optimizer.py

Cost-based query optimizer for adaptive RAG retrieval.

Classifies incoming queries by type (keyword_heavy, semantic, factual) using
lightweight features, then selects the cheapest retrieval plan that meets a
quality threshold. This mirrors how a database query optimizer estimates costs
and selects execution plans.

The lookup table is derived from profiling data produced by
scripts/profile_retrieval_plans.py. It can be loaded from a JSON file or
use built-in defaults.
"""

from __future__ import annotations

import json
import math
import os
import pathlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ── Query feature extraction ──────────────────────────────────────────────

# Common English stopwords for feature extraction
_STOPWORDS = frozenset({
    "the", "is", "at", "which", "on", "for", "a", "an", "and", "or", "in",
    "to", "of", "by", "with", "that", "this", "it", "as", "are", "was",
    "what", "how", "why", "when", "where", "who", "does", "do", "be",
    "can", "could", "would", "should", "will", "have", "has", "had",
    "been", "being", "about", "from", "into", "through", "during", "before",
    "after", "above", "below", "between", "up", "down", "out", "off", "over",
    "under", "again", "further", "then", "once", "me", "my", "i", "you",
    "your", "we", "our", "they", "their", "he", "she", "him", "her",
    "its", "there", "here", "some", "any", "all", "each", "every",
    "both", "few", "more", "most", "other", "such", "no", "not", "only",
    "own", "same", "so", "than", "too", "very", "just", "also",
    "tell", "show", "explain", "describe", "define", "give", "list",
})

# Database-specific technical terms that signal keyword-heavy queries
_DB_KEYWORDS = frozenset({
    "acid", "atomicity", "consistency", "isolation", "durability",
    "transaction", "commit", "rollback", "abort", "log", "wal",
    "aries", "checkpoint", "redo", "undo", "recovery",
    "lock", "latch", "deadlock", "2pl", "two-phase", "mvcc",
    "timestamp", "serializable", "schedule", "conflict",
    "b+", "btree", "b-tree", "hash", "index", "clustered",
    "normalization", "bcnf", "3nf", "1nf", "2nf", "4nf", "5nf",
    "functional", "dependency", "decomposition", "lossless", "lossy",
    "superkey", "candidate", "primary", "foreign", "key",
    "relation", "tuple", "attribute", "schema", "instance",
    "sql", "select", "join", "aggregate", "group", "having",
    "subquery", "view", "trigger", "assertion", "constraint",
    "er", "entity", "relationship", "cardinality", "participation",
    "oltp", "olap", "warehouse", "star", "snowflake",
    "buffer", "page", "disk", "storage", "heap", "sequential",
    "query", "optimizer", "cost", "plan", "execution",
    "concurrency", "parallel", "distributed", "replication",
    "nosql", "column-store", "row-store", "in-memory",
})

# Interrogative words that signal different query types
_FACTUAL_STARTERS = {"what", "who", "which", "name", "list", "define"}
_COMPARISON_STARTERS = {"contrast", "compare", "difference", "versus", "vs"}
_EXPLANATION_STARTERS = {"how", "why", "explain", "describe", "show"}


@dataclass
class QueryFeatures:
    """Lightweight feature vector for a query."""
    token_count: int = 0
    content_token_count: int = 0       # tokens after stopword removal
    keyword_density: float = 0.0       # fraction of tokens that are DB keywords
    avg_term_specificity: float = 0.0  # average IDF-like score of content tokens
    has_comparison_word: bool = False
    has_factual_starter: bool = False
    has_explanation_starter: bool = False
    query_type: str = "semantic"       # classified type


def _normalize_token(token: str) -> str:
    """Cheap morphological normalization: strip common suffixes."""
    # Handle plural/gerund forms so "transactions" matches "transaction"
    if token.endswith("ies"):
        return token[:-3] + "y"   # properties -> property (not a DB keyword, but consistent)
    if token.endswith("ing"):
        return token[:-3]
    if token.endswith("tion"):
        return token               # keep as-is: "transaction", "isolation"
    if token.endswith("s") and len(token) > 3:
        return token[:-1]
    return token


def extract_features(query: str) -> QueryFeatures:
    """
    Extract lightweight features from a query string.
    Runs in <1ms — no model inference needed.
    """
    tokens = query.lower().split()
    cleaned_tokens = [t.strip(".,!?()[]:'\"") for t in tokens]
    cleaned_tokens = [t for t in cleaned_tokens if t]

    content_tokens = [t for t in cleaned_tokens if t not in _STOPWORDS]

    # Keyword density: fraction of content tokens that are known DB terms
    # Use normalized forms so plurals/gerunds match the keyword set
    if content_tokens:
        db_hits = sum(
            1 for t in content_tokens
            if t in _DB_KEYWORDS or _normalize_token(t) in _DB_KEYWORDS
        )
        keyword_density = db_hits / len(content_tokens)
    else:
        keyword_density = 0.0

    # Term specificity: longer, rarer-looking tokens get higher scores
    # This is a cheap proxy for IDF without needing corpus statistics
    specificity_scores = []
    for t in content_tokens:
        # Heuristic: length-based specificity (short common words score low)
        length_score = min(1.0, len(t) / 10.0)
        # Bonus for DB keywords (check both raw and normalized forms)
        is_db = t in _DB_KEYWORDS or _normalize_token(t) in _DB_KEYWORDS
        db_bonus = 0.3 if is_db else 0.0
        specificity_scores.append(length_score + db_bonus)

    avg_specificity = (
        sum(specificity_scores) / len(specificity_scores)
        if specificity_scores else 0.0
    )

    # Detect query intent from first content word
    first_word = cleaned_tokens[0] if cleaned_tokens else ""
    has_comparison = any(w in _COMPARISON_STARTERS for w in cleaned_tokens[:3])
    has_factual = first_word in _FACTUAL_STARTERS
    has_explanation = first_word in _EXPLANATION_STARTERS

    features = QueryFeatures(
        token_count=len(cleaned_tokens),
        content_token_count=len(content_tokens),
        keyword_density=keyword_density,
        avg_term_specificity=avg_specificity,
        has_comparison_word=has_comparison,
        has_factual_starter=has_factual,
        has_explanation_starter=has_explanation,
    )

    # Classify query type
    features.query_type = _classify_query(features)
    return features


def _classify_query(features: QueryFeatures) -> str:
    """
    Rule-based query classifier.

    - keyword_heavy: High density of DB-specific terms (>= 0.4).
      These queries contain enough specific vocabulary that BM25
      keyword matching is highly effective.

    - factual: Short queries starting with "what/who/which" with
      moderate keyword density. Simple lookups where a single
      retriever often suffices.

    - semantic: Everything else — open-ended, comparative, or
      explanatory queries where vector similarity adds the most
      value.
    """
    if features.keyword_density >= 0.4:
        return "keyword_heavy"
    elif features.has_factual_starter and features.token_count <= 10:
        return "factual"
    elif features.has_comparison_word:
        return "semantic"
    elif features.has_explanation_starter and features.keyword_density < 0.2:
        return "semantic"
    elif features.keyword_density >= 0.25:
        return "keyword_heavy"
    else:
        return "semantic"


# ── Plan selection ─────────────────────────────────────────────────────────

@dataclass
class RetrievalPlan:
    """A concrete retrieval execution plan."""
    name: str
    weights: Dict[str, float]
    rerank: bool
    label: str

    def active_retrievers(self) -> List[str]:
        """Names of retrievers with non-zero weight."""
        return [k for k, v in self.weights.items() if v > 0]


# Default lookup table derived from profiling analysis.
# Maps query_type -> plan configuration.
# Key insight: keyword-heavy queries do well with BM25-only (fastest),
# semantic queries need FAISS+BM25, factual queries use FAISS-only.
_DEFAULT_LOOKUP: Dict[str, Dict] = {
    "keyword_heavy": {
        "name": "bm25_only",
        "weights": {"faiss": 0.0, "bm25": 1.0, "index_keywords": 0.0},
        "rerank": False,
        "label": "BM25 Only",
    },
    "semantic": {
        "name": "faiss_bm25",
        "weights": {"faiss": 0.5, "bm25": 0.5, "index_keywords": 0.0},
        "rerank": False,
        "label": "FAISS + BM25",
    },
    "factual": {
        "name": "faiss_only",
        "weights": {"faiss": 1.0, "bm25": 0.0, "index_keywords": 0.0},
        "rerank": False,
        "label": "FAISS Only",
    },
}


class QueryOptimizer:
    """
    Cost-based query optimizer that selects a retrieval plan based on
    query characteristics.

    Analogous to a database query optimizer:
      - extract_features() ~ column statistics / histograms
      - _classify_query()  ~ cardinality estimation
      - select_plan()      ~ plan enumeration + cost-based selection

    Usage:
        optimizer = QueryOptimizer()
        plan = optimizer.select_plan("What are the ACID properties?")
        # plan.weights -> {"faiss": 0.0, "bm25": 1.0, "index_keywords": 0.0}
        # plan.rerank -> False
    """

    def __init__(self, lookup_path: Optional[str] = None):
        """
        Args:
            lookup_path: Path to a lookup_table.json file generated by
                         profile_retrieval_plans.py. If None, uses built-in
                         defaults.
        """
        if lookup_path and os.path.exists(lookup_path):
            self.lookup = self._load_lookup(lookup_path)
        else:
            self.lookup = _DEFAULT_LOOKUP

    @staticmethod
    def _load_lookup(path: str) -> Dict[str, Dict]:
        """Load a profiling-derived lookup table and convert plan names to configs."""
        from scripts.profile_retrieval_plans import PLANS
        with open(path, "r") as f:
            raw = json.load(f)  # {query_type: plan_name}

        lookup = {}
        for qtype, plan_name in raw.items():
            if plan_name in PLANS:
                plan_def = PLANS[plan_name]
                lookup[qtype] = {
                    "name": plan_name,
                    "weights": plan_def["weights"],
                    "rerank": plan_def["rerank"],
                    "label": plan_def["label"],
                }
            else:
                # Fallback to default
                lookup[qtype] = _DEFAULT_LOOKUP.get(qtype, _DEFAULT_LOOKUP["semantic"])

        return lookup

    def select_plan(self, query: str) -> RetrievalPlan:
        """
        Classify a query and return the optimal retrieval plan.

        Args:
            query: The user's question string.

        Returns:
            RetrievalPlan with weights, rerank flag, and label.
        """
        features = extract_features(query)
        plan_cfg = self.lookup.get(features.query_type, self.lookup.get("semantic", _DEFAULT_LOOKUP["semantic"]))

        return RetrievalPlan(
            name=plan_cfg["name"],
            weights=plan_cfg["weights"],
            rerank=plan_cfg["rerank"],
            label=plan_cfg["label"],
        )

    def explain(self, query: str) -> str:
        """Return a human-readable explanation of the plan selection."""
        features = extract_features(query)
        plan = self.select_plan(query)
        active = plan.active_retrievers()

        lines = [
            f"Query: {query}",
            f"  Type: {features.query_type}",
            f"  Features: {features.content_token_count} content tokens, "
            f"keyword_density={features.keyword_density:.2f}, "
            f"specificity={features.avg_term_specificity:.2f}",
            f"  Plan: {plan.label}",
            f"  Retrievers: {', '.join(active)}",
            f"  Reranking: {'yes' if plan.rerank else 'no'}",
        ]
        return "\n".join(lines)
