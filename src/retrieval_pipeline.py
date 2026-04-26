from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from time import perf_counter
from typing import Any, Dict, Iterable, List, Optional, Sequence

from src.config import RAGConfig
from src.generator import format_prompt
from src.query_enhancement import contextualize_query, decompose_complex_query, generate_hypothetical_document
from src.ranking.ranker import EnsembleRanker
from src.retriever import (
    ArtifactBundle,
    BM25Retriever,
    FAISSRetriever,
    IndexKeywordRetriever,
    Retriever,
)


FOLLOW_UP_PATTERN = re.compile(
    r"^(it|they|them|that|those|this|these|he|she|its|their|what about|how about|and what|and how|why is that|why does that)\b",
    re.IGNORECASE,
)
DEFINITION_PATTERN = re.compile(r"^(what is|what are|define|definition of|meaning of)\b", re.IGNORECASE)
PROCEDURAL_PATTERN = re.compile(r"^(how do|how does|what steps|steps of|procedure|algorithm|phases of)\b", re.IGNORECASE)
EXPLANATORY_PATTERN = re.compile(r"\b(explain|why|how does|how do we|why does|how can)\b", re.IGNORECASE)
MULTIPART_PATTERN = re.compile(
    r"(\?.*\?| and | compare | contrast | difference between | advantages and disadvantages | both .* and )",
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


@dataclass
class QueryPlan:
    """Retrieval plan produced by AdaptiveQueryPlanner describing how to execute a query."""

    query_type: str
    resolved_query_type: str
    effective_query: str
    rewritten_query: Optional[str] = None
    sub_queries: List[str] = field(default_factory=list)
    retrieval_mode: str = "flat"
    chunk_weights: Dict[str, float] = field(default_factory=dict)
    section_weights: Dict[str, float] = field(default_factory=dict)
    num_candidates: int = 0
    section_top_k: int = 0
    section_candidate_pool: int = 0
    use_hyde: bool = False
    diversify_sections: bool = False
    max_chunks_per_section: int = 0


@dataclass
class RetrievalTrace:
    """Diagnostic trace capturing scores, latencies, and decisions from a single retrieval run."""

    query_type: str
    resolved_query_type: str
    original_query: str
    effective_query: str
    rewritten_query: Optional[str]
    retrieval_mode: str
    sub_queries: List[str]
    chunk_weights: Dict[str, float]
    section_weights: Dict[str, float]
    route_reason: str
    chunk_scores: Dict[str, Dict[int, float]] = field(default_factory=dict)
    section_scores: Dict[str, Dict[int, float]] = field(default_factory=dict)
    fused_chunk_ids: List[int] = field(default_factory=list)
    fused_chunk_scores: List[float] = field(default_factory=list)
    fused_section_ids: List[int] = field(default_factory=list)
    retrieval_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    chunks_passed_to_generation: int = 0
    prompt_tokens_estimate: int = 0
    page_map: Dict[int, List[int]] = field(default_factory=dict)
    selected_section_paths: List[str] = field(default_factory=list)
    subquery_traces: List[Dict[str, Any]] = field(default_factory=list)


class AdaptiveQueryPlanner:
    """Classifies incoming queries and builds retrieval plans with query-type-aware weights and modes."""

    def __init__(self, cfg: RAGConfig):
        self.cfg = cfg

    def classify(self, query: str, history: Optional[List[Dict[str, str]]] = None) -> tuple[str, str]:
        """Classify a query into a type (e.g. definition, procedural, multi_part) using regex heuristics."""
        stripped = query.strip()
        lowered = stripped.lower()

        if history and FOLLOW_UP_PATTERN.search(lowered):
            return "follow_up", "follow-up reference pattern"
        if MULTIPART_PATTERN.search(lowered) or lowered.count("?") > 1:
            return "multi_part", "multi-clause question"
        if DEFINITION_PATTERN.search(lowered):
            return "definition", "definition pattern"
        if PROCEDURAL_PATTERN.search(lowered):
            return "procedural", "procedural pattern"
        if EXPLANATORY_PATTERN.search(lowered):
            return "explanatory", "explanatory pattern"
        return "other", "default route"

    @staticmethod
    def _normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
        active = {name: float(weight) for name, weight in weights.items() if weight > 0}
        total = sum(active.values()) or 1.0
        return {name: weight / total for name, weight in active.items()}

    def _decompose_query(self, query: str, model_path: str) -> List[str]:
        heuristic_subqueries = self._heuristic_decompose_query(query)
        if len(heuristic_subqueries) > 1:
            raw_subqueries = heuristic_subqueries
        else:
            raw_subqueries = decompose_complex_query(query, model_path)

        cleaned: List[str] = []
        for candidate in raw_subqueries:
            normalized = candidate.strip(" -•\t")
            if normalized and normalized not in cleaned:
                cleaned.append(normalized)
        if not cleaned:
            return [query]
        if query not in cleaned:
            cleaned.append(query)
        return cleaned[: self.cfg.decomposition_max_subqueries]

    @staticmethod
    def _heuristic_decompose_query(query: str) -> List[str]:
        """Split a query into sub-queries using regex patterns for comparisons, advantages, and clause boundaries."""
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

        clauses = [segment.strip(" ?.,;:") for segment in CLAUSE_SPLIT_PATTERN.split(normalized) if segment.strip(" ?.,;:")]
        if len(clauses) > 1:
            rebuilt = [clause if clause.endswith("?") else f"{clause}?" for clause in clauses]
            return rebuilt

        return [normalized]

    def plan(self, query: str, history: Optional[List[Dict[str, str]]] = None) -> tuple[QueryPlan, str]:
        """Build a retrieval plan by classifying the query and selecting weights, mode, and candidates.

        Args:
            query: The user's raw query string.
            history: Optional conversation history for resolving follow-up references.

        Returns:
            A tuple of (QueryPlan, reason_string) where reason_string explains
            the classification route taken.
        """
        if not self.cfg.enable_adaptive_routing:
            return (
                QueryPlan(
                    query_type="configured",
                    resolved_query_type="configured",
                    effective_query=query,
                    retrieval_mode="flat",
                    chunk_weights=self._normalize_weights(self.cfg.ranker_weights),
                    section_weights={},
                    num_candidates=max(self.cfg.num_candidates, self.cfg.top_k),
                    section_top_k=0,
                    section_candidate_pool=0,
                    use_hyde=self.cfg.use_hyde,
                    diversify_sections=False,
                    max_chunks_per_section=self.cfg.top_k,
                ),
                "adaptive routing disabled",
            )

        query_type, reason = self.classify(query, history)
        effective_query = query
        rewritten_query = None
        sub_queries: List[str] = []
        resolved_query_type = query_type

        if query_type == "follow_up" and history:
            rewritten_query = contextualize_query(query, history, self.cfg.gen_model)
            effective_query = rewritten_query or query
            resolved_query_type, _ = self.classify(effective_query, history=None)
            if resolved_query_type == "follow_up":
                resolved_query_type = "explanatory"

        if resolved_query_type == "multi_part":
            sub_queries = self._decompose_query(effective_query, self.cfg.gen_model)
            if len(sub_queries) <= 1:
                sub_queries = [effective_query]

        base_weights = self.cfg.ranker_weights
        if resolved_query_type == "definition":
            chunk_weights = {**base_weights, "faiss": 0.3, "bm25": 0.55, "index_keywords": 0.15}
        elif resolved_query_type in {"procedural", "follow_up"}:
            chunk_weights = {**base_weights, "faiss": 0.5, "bm25": 0.3, "index_keywords": 0.2}
        elif resolved_query_type in {"explanatory", "multi_part", "other"}:
            chunk_weights = {**base_weights, "faiss": 0.55, "bm25": 0.3, "index_keywords": 0.15}
        else:
            chunk_weights = dict(base_weights)

        section_weights = {"faiss": 0.65, "bm25": 0.35}
        if resolved_query_type == "definition":
            section_weights = {"faiss": 0.35, "bm25": 0.65}
            retrieval_mode = "flat"
        elif resolved_query_type in {"explanatory", "procedural", "follow_up", "multi_part"}:
            retrieval_mode = "hierarchical"
        else:
            retrieval_mode = "flat"

        if not self.cfg.enable_hierarchical_retrieval:
            retrieval_mode = "flat"

        if resolved_query_type == "multi_part":
            num_candidates = max(self.cfg.num_candidates, self.cfg.top_k * 6)
            section_top_k = max(self.cfg.section_top_k, 6)
            section_candidate_pool = max(section_top_k * 4, 12)
        elif retrieval_mode == "hierarchical":
            num_candidates = max(self.cfg.num_candidates, self.cfg.top_k * 5)
            section_top_k = max(self.cfg.section_top_k, 4)
            section_candidate_pool = max(section_top_k * 3, 8)
        else:
            num_candidates = max(self.cfg.num_candidates, self.cfg.top_k * 4)
            section_top_k = self.cfg.section_top_k
            section_candidate_pool = max(section_top_k, 1)

        return (
            QueryPlan(
                query_type=query_type,
                resolved_query_type=resolved_query_type,
                effective_query=effective_query,
                rewritten_query=rewritten_query,
                sub_queries=sub_queries,
                retrieval_mode=retrieval_mode,
                chunk_weights=self._normalize_weights(chunk_weights),
                section_weights=self._normalize_weights(section_weights),
                num_candidates=num_candidates,
                section_top_k=section_top_k,
                section_candidate_pool=section_candidate_pool,
                use_hyde=self.cfg.use_hyde and query_type != "follow_up",
                diversify_sections=retrieval_mode == "hierarchical",
                max_chunks_per_section=2 if retrieval_mode == "hierarchical" else self.cfg.top_k,
            ),
            reason,
        )


def build_runtime_retrievers(
    bundle: ArtifactBundle,
    cfg: RAGConfig,
) -> Dict[str, List[Retriever]]:
    """Instantiate chunk and section retrievers from pre-built artifacts and config."""
    chunk_retrievers: List[Retriever] = [
        FAISSRetriever(bundle.chunk_index, cfg.embed_model, bundle.chunk_embeddings),
        BM25Retriever(bundle.chunk_bm25),
    ]
    if cfg.ranker_weights.get("index_keywords", 0) > 0:
        chunk_retrievers.append(
            IndexKeywordRetriever(cfg.extracted_index_path, cfg.page_to_chunk_map_path)
        )

    section_retrievers: List[Retriever] = []
    if bundle.has_hierarchical_artifacts:
        section_retrievers = [
            FAISSRetriever(bundle.section_index, cfg.embed_model, bundle.section_embeddings),
            BM25Retriever(bundle.section_bm25),
        ]

    return {"chunk": chunk_retrievers, "section": section_retrievers}


def _active_weights(weights: Dict[str, float], retrievers: Sequence[Retriever]) -> Dict[str, float]:
    """Filter and re-normalize weights to only include retrievers that are available."""
    available = {retriever.name for retriever in retrievers}
    filtered = {name: weight for name, weight in weights.items() if weight > 0 and name in available}
    total = sum(filtered.values()) or 1.0
    return {name: weight / total for name, weight in filtered.items()}


def _score_with_retrievers(
    *,
    query: str,
    texts: Sequence[str],
    retrievers: Sequence[Retriever],
    weights: Dict[str, float],
    pool_size: int,
    candidate_ids: Optional[Iterable[int]] = None,
) -> tuple[Dict[str, Dict[int, float]], List[int], List[float]]:
    """Score candidates with all retrievers and fuse results via reciprocal rank fusion."""
    raw_scores: Dict[str, Dict[int, float]] = {}
    for retriever in retrievers:
        raw_scores[retriever.name] = retriever.get_scores(query, pool_size, texts, candidate_ids=candidate_ids)

    ranker = EnsembleRanker(
        ensemble_method="rrf",
        weights=_active_weights(weights, retrievers),
        rrf_k=60,
    )
    ordered_ids, ordered_scores = ranker.rank(raw_scores=raw_scores)
    return raw_scores, ordered_ids, ordered_scores


def _lexical_overlap(query: str, text: str) -> float:
    """Compute the fraction of query tokens that appear in the text (case-insensitive)."""
    query_tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
    text_tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
    if not query_tokens:
        return 0.0
    return len(query_tokens & text_tokens) / len(query_tokens)


def _page_specificity(page_numbers: Sequence[int]) -> float:
    """Score how localized a chunk is across pages (1.0 for single-page, decays with spread)."""
    if not page_numbers:
        return 0.0
    if len(page_numbers) == 1:
        return 1.0
    return 1.0 / (1 + (max(page_numbers) - min(page_numbers)))


def _page_aware_rerank(
    *,
    query: str,
    candidate_ids: Sequence[int],
    candidate_scores: Sequence[float],
    metadata: Sequence[Dict[str, Any]],
    chunks: Sequence[str],
) -> tuple[int, ...]:
    """Re-rank candidates using a weighted blend of base score, lexical overlap, and page specificity."""
    if not candidate_ids:
        return ()

    max_score = max(candidate_scores) or 1.0
    scored = [
        (
            cid,
            (0.6 * base / max_score)
            + (0.15 * _lexical_overlap(query, metadata[cid].get("raw_text", chunks[cid])))
            + (0.25 * _page_specificity(metadata[cid].get("page_numbers", []))),
            base,
        )
        for cid, base in zip(candidate_ids, candidate_scores)
    ]
    return tuple(cid for cid, _, _ in sorted(scored, key=lambda t: (t[1], t[2]), reverse=True))


def _score_lookup(candidate_ids: Sequence[int], candidate_scores: Sequence[float]) -> Dict[int, float]:
    """Build a normalized score lookup dict mapping candidate IDs to scores in [0, 1]."""
    if not candidate_ids or not candidate_scores:
        return {}
    max_score = max(candidate_scores) or 1.0
    return {
        int(candidate_id): float(score) / max_score
        for candidate_id, score in zip(candidate_ids, candidate_scores)
    }


def _below_confidence_threshold(scores: Sequence[float], threshold: float) -> bool:
    """Return true when a result list is empty or its best fused score is below threshold."""
    if threshold <= 0:
        return False
    return not scores or max(scores) < threshold


def _apply_section_prior(
    candidate_ids: Sequence[int],
    candidate_scores: Sequence[float],
    metadata: Sequence[Dict[str, Any]],
    section_score_lookup: Dict[int, float],
) -> tuple[List[int], List[float]]:
    """Boost chunk scores by blending in their parent section's retrieval score."""
    if not candidate_ids:
        return [], []

    normalized_lookup = _score_lookup(candidate_ids, candidate_scores)
    rescored = []
    for candidate_id in candidate_ids:
        section_id = metadata[candidate_id].get("section_id")
        section_prior = section_score_lookup.get(int(section_id), 0.0) if section_id is not None else 0.0
        combined_score = (0.8 * normalized_lookup.get(candidate_id, 0.0)) + (0.2 * section_prior)
        rescored.append((candidate_id, combined_score))

    rescored.sort(key=lambda item: item[1], reverse=True)
    return [candidate_id for candidate_id, _ in rescored], [score for _, score in rescored]


def _collect_section_candidate_ids(section_ids: Sequence[int], bundle: ArtifactBundle) -> List[int]:
    """Gather deduplicated chunk IDs belonging to the given sections."""
    candidate_ids = []
    seen = set()
    for section_id in section_ids:
        for chunk_id in bundle.section_meta[section_id].get("chunk_ids", []):
            if chunk_id not in seen:
                candidate_ids.append(int(chunk_id))
                seen.add(int(chunk_id))
    return candidate_ids


def _merge_multi_part_results(results: List[Dict[str, Any]]) -> tuple[List[int], List[float]]:
    """Merge retrieval results from multiple sub-queries using score + rank fusion with coverage bonuses."""
    merged_scores: Dict[int, float] = defaultdict(float)
    coverage: Dict[int, int] = defaultdict(int)
    best_rank: Dict[int, int] = {}
    for result in results:
        candidate_ids = result.get("candidate_ids", [])
        candidate_scores = result.get("candidate_scores", [])
        normalized_lookup = _score_lookup(candidate_ids, candidate_scores)
        for rank, candidate_id in enumerate(candidate_ids, start=1):
            score = normalized_lookup.get(candidate_id, 0.0)
            merged_scores[candidate_id] += (0.65 * score) + (0.35 / (rank + 4))
            coverage[candidate_id] += 1
            best_rank[candidate_id] = min(best_rank.get(candidate_id, rank), rank)

    ranked = sorted(
        merged_scores.items(),
        key=lambda item: (
            item[1] + (0.1 * coverage[item[0]]),
            coverage[item[0]],
            -best_rank.get(item[0], 10**9),
        ),
        reverse=True,
    )
    return [candidate_id for candidate_id, _ in ranked], [score for _, score in ranked]


def _diversify_by_section(
    candidate_ids: Sequence[int],
    metadata: Sequence[Dict[str, Any]],
    limit: int,
    max_per_section: int,
) -> List[int]:
    """Select up to limit candidates while capping per-section representation for diversity."""
    if not candidate_ids:
        return []

    selected: List[int] = []
    section_counts: Dict[Optional[int], int] = defaultdict(int)
    deferred: List[int] = []

    for candidate_id in candidate_ids:
        section_id = metadata[candidate_id].get("section_id")
        if section_counts[section_id] < max_per_section:
            selected.append(candidate_id)
            section_counts[section_id] += 1
            if len(selected) == limit:
                return selected
        else:
            deferred.append(candidate_id)

    for candidate_id in deferred:
        if candidate_id not in selected:
            selected.append(candidate_id)
            if len(selected) == limit:
                break

    return selected


def _estimate_prompt_tokens(query: str, chunks: Sequence[str], prompt_mode: str) -> int:
    """Estimate the token count of the final generation prompt from the assembled chunks."""
    prompt = format_prompt(list(chunks), query, system_prompt_mode=prompt_mode)
    return max(1, round(len(prompt.split()) * 1.25))


def execute_retrieval_plan(
    *,
    query: str,
    cfg: RAGConfig,
    bundle: ArtifactBundle,
    retrievers: Dict[str, List[Retriever]],
    history: Optional[List[Dict[str, str]]] = None,
) -> tuple[List[str], List[int], RetrievalTrace]:
    """Plan and execute end-to-end retrieval: classify, score, rerank, and select final chunks.

    Args:
        query: The user's raw query string.
        cfg: RAG configuration controlling top-k, weights, and feature flags.
        bundle: Pre-built artifact bundle containing chunks, sections, indices, and metadata.
        retrievers: Dict mapping "chunk" and "section" to their respective retriever lists.
        history: Optional conversation history for follow-up resolution.

    Returns:
        A tuple of (ranked_chunks, chunk_ids, trace) where ranked_chunks are the
        selected text passages, chunk_ids are their indices, and trace is a
        RetrievalTrace with full diagnostic information.
    """
    planner = AdaptiveQueryPlanner(cfg)
    plan, route_reason = planner.plan(query, history=history)
    retrieval_query = plan.effective_query
    retrieval_start = perf_counter()

    merged_section_scores: Dict[str, Dict[int, float]] = {}
    selected_section_ids: List[int] = []
    merged_raw_chunk_scores: Dict[str, Dict[int, float]] = {}
    ordered_chunk_ids: List[int] = []
    ordered_chunk_scores: List[float] = []
    query_inputs = plan.sub_queries or [retrieval_query]

    if plan.use_hyde:
        retrieval_query = generate_hypothetical_document(
            retrieval_query,
            cfg.gen_model,
            max_tokens=cfg.hyde_max_tokens,
        )
        query_inputs = [retrieval_query]

    multi_part_results: List[Dict[str, Any]] = []
    section_score_lookup: Dict[int, float] = {}
    subquery_traces: List[Dict[str, Any]] = []

    for query_input in query_inputs:
        candidate_ids = None
        subquery_section_ids: List[int] = []
        subquery_section_lookup: Dict[int, float] = {}
        if plan.retrieval_mode == "hierarchical" and bundle.has_hierarchical_artifacts and retrievers["section"]:
            section_scores, ordered_sections, section_ordered_scores = _score_with_retrievers(
                query=query_input,
                texts=bundle.sections,
                retrievers=retrievers["section"],
                weights=plan.section_weights,
                pool_size=max(plan.section_candidate_pool, 1),
            )
            for retriever_name, scores in section_scores.items():
                merged_section_scores.setdefault(retriever_name, {}).update(scores)
            subquery_section_ids = ordered_sections[: plan.section_top_k]
            subquery_section_lookup = _score_lookup(ordered_sections, section_ordered_scores)
            candidate_ids = _collect_section_candidate_ids(ordered_sections[: plan.section_candidate_pool], bundle)
            for section_id, score in subquery_section_lookup.items():
                section_score_lookup[section_id] = max(section_score_lookup.get(section_id, 0.0), score)
            for section_id in subquery_section_ids:
                if section_id not in selected_section_ids:
                    selected_section_ids.append(section_id)

        raw_chunk_scores, chunk_ids, chunk_scores = _score_with_retrievers(
            query=query_input,
            texts=bundle.chunks,
            retrievers=retrievers["chunk"],
            weights=plan.chunk_weights,
            pool_size=plan.num_candidates,
            candidate_ids=candidate_ids,
        )

        if _below_confidence_threshold(chunk_scores, cfg.retrieval_confidence_threshold):
            fallback_scores, fallback_ids, fallback_ordered_scores = _score_with_retrievers(
                query=query_input,
                texts=bundle.chunks,
                retrievers=retrievers["chunk"],
                weights=plan.chunk_weights,
                pool_size=plan.num_candidates * cfg.fallback_candidate_multiplier,
                candidate_ids=None,
            )
            if fallback_ordered_scores and (
                not chunk_scores or max(fallback_ordered_scores) >= max(chunk_scores)
            ):
                raw_chunk_scores = fallback_scores
                chunk_ids = fallback_ids
                chunk_scores = fallback_ordered_scores
                candidate_ids = None

        for retriever_name, scores in raw_chunk_scores.items():
            merged_raw_chunk_scores.setdefault(retriever_name, {}).update(scores)

        if subquery_section_lookup:
            chunk_ids, chunk_scores = _apply_section_prior(
                chunk_ids,
                chunk_scores,
                bundle.metadata,
                subquery_section_lookup,
            )

        subquery_traces.append(
            {
                "query": query_input,
                "section_ids": subquery_section_ids,
                "candidate_chunk_ids": chunk_ids[: min(len(chunk_ids), cfg.top_k * 3)],
            }
        )

        if plan.query_type == "multi_part":
            multi_part_results.append(
                {
                    "query": query_input,
                    "candidate_ids": chunk_ids,
                    "candidate_scores": chunk_scores,
                }
            )
        else:
            ordered_chunk_ids = chunk_ids
            ordered_chunk_scores = chunk_scores

    if plan.query_type == "multi_part":
        ordered_chunk_ids, ordered_chunk_scores = _merge_multi_part_results(multi_part_results)

    reranked_chunk_ids = _page_aware_rerank(
        query=plan.effective_query,
        candidate_ids=ordered_chunk_ids[: cfg.page_rerank_window],
        candidate_scores=ordered_chunk_scores[: cfg.page_rerank_window],
        metadata=bundle.metadata,
        chunks=bundle.chunks,
    )
    reranked_set = set(reranked_chunk_ids)
    remaining_ids = [candidate_id for candidate_id in ordered_chunk_ids if candidate_id not in reranked_set]
    preselected_chunk_ids = list(reranked_chunk_ids) + remaining_ids
    if plan.diversify_sections:
        final_chunk_ids = _diversify_by_section(
            preselected_chunk_ids,
            bundle.metadata,
            cfg.top_k,
            max(1, plan.max_chunks_per_section),
        )
    else:
        final_chunk_ids = preselected_chunk_ids[: cfg.top_k]
    ranked_chunks = [bundle.chunks[candidate_id] for candidate_id in final_chunk_ids]
    score_lookup = dict(zip(ordered_chunk_ids, ordered_chunk_scores))
    selected_section_paths = [
        bundle.section_meta[section_id].get("section_path", "")
        for section_id in selected_section_ids
        if 0 <= section_id < len(bundle.section_meta)
    ]

    retrieval_latency_ms = (perf_counter() - retrieval_start) * 1000
    trace = RetrievalTrace(
        query_type=plan.query_type,
        resolved_query_type=plan.resolved_query_type,
        original_query=query,
        effective_query=plan.effective_query,
        rewritten_query=plan.rewritten_query,
        retrieval_mode=plan.retrieval_mode,
        sub_queries=query_inputs if plan.query_type == "multi_part" else [],
        chunk_weights=plan.chunk_weights,
        section_weights=plan.section_weights,
        route_reason=route_reason,
        chunk_scores=merged_raw_chunk_scores,
        section_scores=merged_section_scores,
        fused_chunk_ids=final_chunk_ids,
        fused_chunk_scores=[float(score_lookup.get(candidate_id, 0.0)) for candidate_id in final_chunk_ids],
        fused_section_ids=selected_section_ids,
        retrieval_latency_ms=retrieval_latency_ms,
        total_latency_ms=retrieval_latency_ms,
        chunks_passed_to_generation=len(ranked_chunks),
        prompt_tokens_estimate=_estimate_prompt_tokens(plan.effective_query, ranked_chunks, cfg.system_prompt_mode),
        page_map={candidate_id: bundle.metadata[candidate_id].get("page_numbers", []) for candidate_id in final_chunk_ids},
        selected_section_paths=selected_section_paths,
        subquery_traces=subquery_traces,
    )
    return ranked_chunks, final_chunk_ids, trace


def trace_to_dict(trace: RetrievalTrace) -> Dict[str, Any]:
    """Convert a RetrievalTrace dataclass to a plain dictionary for serialization."""
    return asdict(trace)
