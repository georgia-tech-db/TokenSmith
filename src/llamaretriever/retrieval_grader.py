"""
Non-LLM retrieval grading and filtering.

Reduces obviously weak retrievals before agent curation while preserving
enough recall to avoid over-pruning.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from llama_index.core.schema import NodeWithScore

from .config import LlamaIndexConfig


_STOPWORDS = frozenset({
    "a", "an", "the", "in", "on", "at", "to", "for", "of", "with", "by",
    "from", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall",
    "and", "or", "but", "not", "so", "if", "then", "than",
    "this", "that", "these", "those", "it", "its",
    "what", "which", "who", "whom", "whose", "where", "when", "why", "how",
})


@dataclass
class GradedNode:
    node: NodeWithScore
    final_score: float
    keyword_hits: int
    matched_terms: list[str]


def extract_query_terms(question: str) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9_+-]*", question)
    out: list[str] = []
    seen: set[str] = set()
    for w in words:
        lw = w.lower()
        if lw in _STOPWORDS or len(lw) < 3 or lw in seen:
            continue
        seen.add(lw)
        out.append(w)
    return out


def _text_blob(node: NodeWithScore) -> str:
    md = node.metadata or {}
    parts = [
        md.get("chapter", ""),
        md.get("section", ""),
        md.get("subsection", ""),
        md.get("header_path", ""),
        md.get("raw_text", "") or node.text or "",
    ]
    return "\n".join(parts).lower()


def grade_retrieved_nodes(
    nodes: list[NodeWithScore],
    question: str,
    cfg: LlamaIndexConfig,
) -> list[NodeWithScore]:
    terms = extract_query_terms(question)
    graded: list[GradedNode] = []

    for nws in nodes:
        raw_score = float(nws.score or 0.0)
        blob = _text_blob(nws)

        matched = [t for t in terms if t.lower() in blob]
        keyword_hits = len(matched)

        final_score = raw_score + 0.06 * keyword_hits

        graded.append(
            GradedNode(
                node=nws,
                final_score=final_score,
                keyword_hits=keyword_hits,
                matched_terms=matched,
            )
        )

    graded.sort(key=lambda x: x.final_score, reverse=True)

    kept: list[GradedNode] = []
    for g in graded:
        if (
            g.final_score >= cfg.retrieval_min_score
            and g.keyword_hits >= cfg.retrieval_min_keyword_hits
        ):
            kept.append(g)

    if len(kept) < cfg.retrieval_keep_at_least:
        kept = graded[: min(cfg.retrieval_keep_at_least, len(graded))]

    kept = kept[: cfg.retrieval_max_after_grade]

    out: list[NodeWithScore] = []
    for g in kept:
        g.node.metadata["retrieval_grade_score"] = round(g.final_score, 4)
        g.node.metadata["retrieval_keyword_hits"] = g.keyword_hits
        g.node.metadata["retrieval_matched_terms"] = g.matched_terms
        out.append(g.node)

    return out
