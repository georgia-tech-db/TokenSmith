from __future__ import annotations

import re
from dataclasses import dataclass


QUERY_TYPE_DEFINITION = "definition"
QUERY_TYPE_EXPLANATORY = "explanatory"
QUERY_TYPE_FOLLOW_UP = "follow_up"
QUERY_TYPE_MULTI_PART = "multi_part"
QUERY_TYPE_OTHER = "other"
QUERY_TYPE_PROCEDURAL = "procedural"

ORDINAL_FOLLOW_UP_REFERENCES = (
    ("the first one", 0),
    ("the second one", 1),
    ("the third one", 2),
    ("the fourth one", 3),
)
FINAL_FOLLOW_UP_REFERENCES = ("the last one", "the final one")


@dataclass(frozen=True)
class RouteDecision:
    """Classification result with the reason exposed in retrieval traces."""

    query_type: str
    reason: str


FOLLOW_UP_REFERENCE_PATTERN = re.compile(
    r"\b("
    r"it|they|them|that|those|this|these|he|she|its|their|"
    + "|".join(re.escape(phrase) for phrase, _ in ORDINAL_FOLLOW_UP_REFERENCES)
    + "|"
    + "|".join(re.escape(phrase) for phrase in FINAL_FOLLOW_UP_REFERENCES)
    + "|"
    r"such\s+(?:a|an|the)?\s*[a-z][a-z0-9-]*|"
    r"what about|how about|and what|and how|why is that|why does that"
    r")\b",
    re.IGNORECASE,
)
DEFINITION_PATTERN = re.compile(r"^(what is|what are|define|definition of|meaning of)\b", re.IGNORECASE)
PROCEDURAL_PATTERN = re.compile(
    r"^(how do|how does|how to|what steps|steps of|procedure|algorithm|phases? of)\b",
    re.IGNORECASE,
)
EXPLANATORY_PATTERN = re.compile(r"\b(explain|why|how does|how do we|why does|how can)\b", re.IGNORECASE)
MULTIPART_PATTERN = re.compile(
    r"(\?.*\?|\bcompare\b|\bcontrast\b|\bdifference between\b|"
    r"\badvantages and disadvantages\b|\bboth\b.*\band\b|,\s*and\s+)",
    re.IGNORECASE,
)
COMPARISON_PATTERN = re.compile(
    r"(?:compare|contrast|difference between|differentiate|versus|vs\.?)\s+(?P<left>.+?)\s+(?:and|vs\.?|versus)\s+(?P<right>.+)",
    re.IGNORECASE,
)
ADVANTAGE_PATTERN = re.compile(
    r"advantages?\s+and\s+disadvantages?\s+of\s+(?P<topic>.+)",
    re.IGNORECASE,
)
CLAUSE_SPLIT_PATTERN = re.compile(r"\?\s*|;\s*|\s+(?:and|also)\s+(?=(?:what|why|how|when|where|which)\b)", re.IGNORECASE)


def classify_query(query: str, *, has_history: bool = False) -> RouteDecision:
    """Classify a query into the adaptive retrieval route used by TokenSmith."""
    stripped = query.strip()
    lowered = stripped.lower()

    if has_history and FOLLOW_UP_REFERENCE_PATTERN.search(lowered):
        return RouteDecision(QUERY_TYPE_FOLLOW_UP, "follow-up reference pattern")
    if MULTIPART_PATTERN.search(lowered) or lowered.count("?") > 1:
        return RouteDecision(QUERY_TYPE_MULTI_PART, "multi-clause question")
    if DEFINITION_PATTERN.search(lowered):
        return RouteDecision(QUERY_TYPE_DEFINITION, "definition pattern")
    if PROCEDURAL_PATTERN.search(lowered):
        return RouteDecision(QUERY_TYPE_PROCEDURAL, "procedural pattern")
    if EXPLANATORY_PATTERN.search(lowered):
        return RouteDecision(QUERY_TYPE_EXPLANATORY, "explanatory pattern")
    return RouteDecision(QUERY_TYPE_OTHER, "default route")


def heuristic_decompose_query(query: str) -> list[str]:
    """Split a complex query using deterministic comparison and clause rules."""
    normalized = " ".join(query.strip().split())
    if not normalized:
        return []

    comparison_match = COMPARISON_PATTERN.search(normalized)
    if comparison_match:
        left = comparison_match.group("left").strip(" ?.,;:")
        right = comparison_match.group("right").strip(" ?.,;:")
        return [
            f"What is {left}?",
            f"What is {right}?",
            f"How do {left} and {right} differ?",
        ]

    advantage_match = ADVANTAGE_PATTERN.search(normalized)
    if advantage_match:
        topic = advantage_match.group("topic").strip(" ?.,;:")
        return [
            f"What are the advantages of {topic}?",
            f"What are the disadvantages of {topic}?",
        ]

    clauses = [
        segment.strip(" ?.,;:")
        for segment in CLAUSE_SPLIT_PATTERN.split(normalized)
        if segment.strip(" ?.,;:")
    ]
    if len(clauses) > 1:
        return [clause if clause.endswith("?") else f"{clause}?" for clause in clauses]

    return [normalized]


def should_apply_anchor_rerank(query: str) -> bool:
    """Avoid acronym anchoring when the sub-query asks for a relationship."""
    lowered = query.lower()
    return not any(marker in lowered for marker in (" differ", " compare", " contrast", " versus", " vs."))
