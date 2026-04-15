"""
reranker.py

This module supports re-ranking strategies applied before the generative LLM call.
"""

import re
from typing import Dict, List, Optional
from sentence_transformers import CrossEncoder

# -------------------------- Cross-Encoder Cache --------------------------
_CROSS_ENCODER_CACHE: Dict[str, CrossEncoder] = {}

def get_cross_encoder(model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2"):
    """
    Fetch the cached cross-encoder model to prevent reloading on every query.
    """
    if model_name not in _CROSS_ENCODER_CACHE:
        _CROSS_ENCODER_CACHE[model_name] = CrossEncoder(model_name)
    return _CROSS_ENCODER_CACHE[model_name]


# -------------------------- Reranking Strategies -------------------------


def _min_max_normalize(values: List[float], neutral: float = 0.5) -> List[float]:
    if not values:
        return []
    v_min = min(values)
    v_max = max(values)
    if v_max - v_min < 1e-12:
        return [neutral] * len(values)
    return [(v - v_min) / (v_max - v_min) for v in values]


def _remove_duplicates_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        key = item.lower()
        if key not in seen:
            out.append(item)
            seen.add(key)
    return out


def _split_coord_list(text: str) -> List[str]:
    parts = re.split(r"\s*(?:,|\band\b|\bor\b)\s*", text, flags=re.IGNORECASE)
    cleaned = []
    for part in parts:
        p = re.sub(r"\s+", " ", part.strip(" .?!:"))
        p = re.sub(r"^(?:and|or)\s+", "", p, flags=re.IGNORECASE)
        if p:
            cleaned.append(p)
    return _remove_duplicates_preserve_order(cleaned)


def _looks_like_complete_subquery(text: str) -> bool:
    """
    Lightweight quality gate to avoid fragmentary fallback subqueries.
    """
    t = text.strip().lower()
    if len(re.findall(r"\w+", t)) < 3:
        return False

    if re.match(r"^(how|why|what|which|when|where|is|are|does|do|can|should)\b", t):
        return True

    return bool(re.search(
        r"\b(is|are|was|were|be|being|been|do|does|did|can|could|should|would|"
        r"may|might|must|has|have|had|influence|affect|impact|prevent|cause|"
        r"improve|optimize|balance|compare|contrast|interact|relate|"
        r"explain|describe|discuss|analyze|evaluate)\b",
        t,
    ))


def _merge_weak_fragments(parts: List[str]) -> List[str]:
    """
    For weaker clauses that are identified as subqueries, merge with neighboring subqueries for more completeness.
    """
    merged = [p for p in parts if p]
    if not merged:
        return []

    i = 0
    while i < len(merged):
        if _looks_like_complete_subquery(merged[i]):
            i += 1
            continue

        # If the next part starts a new imperative clause, merge this weak bridge with the previous part
        starts_with_imperative_cue = bool(re.match(r"^(explain|describe|discuss|analyze|evaluate|compare|contrast|summarize)\b", merged[i+1].strip().lower()))
        if i > 0 and i + 1 < len(merged) and starts_with_imperative_cue:
            merged[i - 1] = f"{merged[i - 1]} and {merged[i]}".strip()
            del merged[i]
            i -= 1
            continue

        if i + 1 < len(merged):
            # otherwise merge weak prefix into the following clause
            merged[i + 1] = f"{merged[i]} {merged[i + 1]}".strip()
            del merged[i]
            continue

        if i > 0:
            # trailing weak fragment can merge into previous clause
            merged[i - 1] = f"{merged[i - 1]} {merged[i]}".strip()
            del merged[i]
            i -= 1
            continue
        i += 1

    return _remove_duplicates_preserve_order([re.sub(r"\s+", " ", p).strip() for p in merged if p.strip()])


def _structured_cartesian_subqueries(query: str) -> List[str]:
    """
    expanding "how do A and B influence C and D" by generating cross-product subqueries.
    """
    q = re.sub(r"\s+", " ", query.strip()).rstrip(".?!")
    if not q:
        return []

    patterns_caught = r"(influence|affect|impact|shape|determine|interact with|relate to)"
    match = re.search(rf"^(?P<left>.+?)\s+(?P<verb>{patterns_caught})\s+(?P<right>.+)$", q, flags=re.IGNORECASE)
    if not match:
        return []

    left_raw = match.group("left").strip()
    verb = match.group("verb").strip().lower()
    right_raw = match.group("right").strip()

    context = ""
    context_match = re.match(r"^(?P<context>.*?),\s*how\s+(?:do|does|can|should|would)\s+.+$", q, flags=re.IGNORECASE)
    if context_match:
        context = context_match.group("context").strip()

    # left_factors
    left_factors = re.sub(r"^.*?\bhow\s+(?:do|does|can|should|would)\s+", "", left_raw, flags=re.IGNORECASE).strip()
    if not left_factors:
        left_factors = left_raw

    left_items = _split_coord_list(left_factors)
    right_items = _split_coord_list(right_raw)

    if len(left_items) < 2 and len(right_items) < 2:
        return []

    subqueries = []
    for l_item in left_items:
        for r_item in right_items:
            if context:
                subqueries.append(f"{context}, how does {l_item} {verb} {r_item}")
            else:
                subqueries.append(f"how does {l_item} {verb} {r_item}")
    return _remove_duplicates_preserve_order(subqueries)


def _structured_compare_subqueries(query: str) -> List[str]:
    """
    expanding "Compare A, B, and C: which X, and how Y?" subqueries.
    """
    q = re.sub(r"\s+", " ", query.strip()).rstrip(".?!")
    if not re.match(r"^(compare|contrast)\b", q, flags=re.IGNORECASE):
        return []

    # identify compared items, criteria
    q_body = re.sub(r"^(compare|contrast)\s+", "", q, flags=re.IGNORECASE).strip()
    if re.search(r"\bon\b", q_body, flags=re.IGNORECASE):
        left_raw, right_raw = re.split(r"\bon\b", q_body, maxsplit=1, flags=re.IGNORECASE)
    elif ":" in q_body:
        left_raw, right_raw = q_body.split(":", 1)
    else:
        # fallback, more flexible pattern
        match = re.match(r"^(?P<left>.+?)\s+(?P<right>(which|how|what).+)$", q_body, flags=re.IGNORECASE)
        if not match:
            return []
        left_raw = match.group("left")
        right_raw = match.group("right")

    items = _split_coord_list(left_raw)
    criteria = _split_coord_list(right_raw)

    criteria = [c for c in criteria if len(re.findall(r"\w+", c)) >= 1]
    if len(items) < 2 or not criteria:
        return []

    subqueries = []
    for item in items:
        for criterion in criteria:
            if re.search(r"^(which|how|what)\b", criterion, flags=re.IGNORECASE):
                subqueries.append(f"For {item}, {criterion}")
            else:
                subqueries.append(f"For {item}, {criterion}")
    return _remove_duplicates_preserve_order(subqueries)


def _structured_balance_subqueries(query: str) -> List[str]:
    """
    expanding "how should A, B, and C be balanced/prioritized/optimized?"
    """
    q = re.sub(r"\s+", " ", query.strip()).rstrip(".?!")
    match = re.match(r"^(?:(?P<context>.+?),\s*)?how\s+should\s+(?P<items>.+?)\s+be\s+(?P<objective>balanced|prioritized|optimised|optimized|weighted|allocated)$", q, flags=re.IGNORECASE)

    if not match:
        return []

    context = (match.group("context") or "").strip()
    objective = match.group("objective").strip().lower()
    items = _split_coord_list(match.group("items").strip())
    if len(items) < 2:
        return []
    if objective == "optimised":
        objective = "optimized"

    subqueries = []
    for item in items:
        if context:
            subqueries.append(f"{context}, how should {item} be {objective}?")
        else:
            subqueries.append(f"How should {item} be {objective}?")
    return _remove_duplicates_preserve_order(subqueries)


def _structured_binary_choice_subqueries(query: str) -> List[str]:
    """
    expand patterns- "Is A or B better as/for X?" "Which is better, A or B, for X?"
    """
    q = re.sub(r"\s+", " ", query.strip()).rstrip(".?!")
    if not q:
        return []

    match = re.match(
        r"^(?:is|are)\s+(?P<a>.+?)\s+or\s+(?P<b>.+?)\s+better\s+"
        r"(?:as|for)\s+(?P<criterion>.+)$",
        q,
        flags=re.IGNORECASE,
    )
    if not match:
        match = re.match(
            r"^which\s+is\s+better,\s*(?P<a>.+?)\s+or\s+(?P<b>.+?),\s*"
            r"(?:as|for)\s+(?P<criterion>.+)$",
            q,
            flags=re.IGNORECASE,
        )
    if not match:
        return []

    left = match.group("a").strip(" ,")
    right = match.group("b").strip(" ,")
    criterion = match.group("criterion").strip()

    if not left or not right or not criterion:
        return []

    return [f"How good is {left} as {criterion}?", f"How good is {right} as {criterion}?"]


def _split_subqueries(query: str) -> List[str]:
    q = re.sub(r"\s+", " ", query.strip())
    if not q:
        return []

    compare_subs = _structured_compare_subqueries(q)
    if compare_subs:
        return compare_subs

    binary_choice_subs = _structured_binary_choice_subqueries(q)
    if binary_choice_subs:
        return binary_choice_subs

    balance_subs = _structured_balance_subqueries(q)
    if balance_subs:
        return balance_subs

    cartesian_subs = _structured_cartesian_subqueries(q)
    if cartesian_subs:
        return cartesian_subs

    normalized = re.sub(r"\b(vs\.?|versus)\b", " and ", q, flags=re.IGNORECASE)

    # fallback: split on conjunctions
    coarse_parts = re.split(
        r"\s*(?:;|,|\band\b|\bor\b|\bwhile\b|\bwhereas\b|\bthen\b)\s*",
        normalized,
        flags=re.IGNORECASE,
    )

    cleaned = []
    for part in coarse_parts:
        p = re.sub(r"\s+", " ", part.strip(" .?!:"))
        p = re.sub(r"^(?:and|or)\s+", "", p, flags=re.IGNORECASE)
        if p:
            cleaned.append(p)

    cleaned = _merge_weak_fragments(cleaned)
    if len(cleaned) >= 2:
        return cleaned

    # second pass
    parts = re.split(
        r"\s*(?:;|,|\bwhile\b|\bwhereas\b|\bthen\b)\s*",
        normalized,
        flags=re.IGNORECASE,
    )

    cleaned = []
    for part in parts:
        p = part.strip(" .?!:")
        if p:
            cleaned.append(p)
    return _merge_weak_fragments(cleaned)


def _is_multi_hop_query(query: str) -> bool:
    """
    Check query based on clause count and presence of markers
    """
    q = query.strip().lower()
    if not q:
        return False

    marker_count = 0
    markers = [
        " and ", " or ", " while ", " whereas ", " compare ", " contrast ",
        " difference between ", " tradeoff", " trade-off", " impact ", " interact ", " relation ", " versus ", " vs ",
    ]
    for marker in markers:
        if marker in f" {q} ":
            marker_count += 1

    num_clauses = len(_split_subqueries(query))
    return num_clauses >= 2 and marker_count >= 1


def _rank_indices_with_cross_encoder(query: str, chunks: List[str]) -> List[int]:
    if not chunks:
        print("No chunks to rerank")
        return []

    model = get_cross_encoder()
    pairs = [(query, chunk) for chunk in chunks]
    scores = [float(x) for x in model.predict(pairs, show_progress_bar=False)]
    return sorted(range(len(chunks)), key=lambda x: scores[x], reverse=True)


def _rank_indices_with_subquery_coverage_details(query: str, chunks: List[str]):
    """
    Returns:
      ranked_indices: list[int]
      final_scores: list[float]
      sub_scores: list[list[float]] where sub_scores[i][j] is score of chunk j for subquery i
    """
    subqueries = _split_subqueries(query)
    if len(subqueries) < 2:
        ranked = _rank_indices_with_cross_encoder(query, chunks)
        return ranked, [], []

    model = get_cross_encoder()
    all_queries = [query] + subqueries
    n_chunks = len(chunks)

    pairs = [(q, chunk) for q in all_queries for chunk in chunks]
    raw_scores = model.predict(pairs, show_progress_bar=False)
    raw_scores = [float(x) for x in raw_scores]

    per_query_scores: List[List[float]] = []
    for q_idx in range(len(all_queries)):
        start = q_idx * n_chunks
        end = start + n_chunks
        per_query_scores.append(_min_max_normalize(raw_scores[start:end]))

    s_full = per_query_scores[0]
    sub_scores = per_query_scores[1:]
    m = len(sub_scores)

    coverage = [max(scores) if scores else 0.0 for scores in sub_scores]

    eps = 1e-6
    gamma = 1.2
    weight_raw = [1.0 / ((eps + cov) ** gamma) for cov in coverage]
    total_weights = sum(weight_raw) or 1.0
    weights = [w / total_weights for w in weight_raw]

    alpha, beta, delta, lmda, tau = 0.65, 0.20, 0.15, 0.35, 0.55

    final_scores: List[float] = []
    for chunk in range(n_chunks):
        sub_chunk_scores = [sub_scores[i][chunk] for i in range(m)]
        weighted_coverage = sum(weights[i] * sub_chunk_scores[i] for i in range(m))
        min_coverage = min(sub_chunk_scores) if sub_chunk_scores else 0.0
        breadth = sum(1 for score in sub_chunk_scores if score >= tau) / m if m else 0.0
        coverage_component = alpha * weighted_coverage + beta * min_coverage + delta * breadth
        score = lmda * s_full[chunk] + (1.0 - lmda) * coverage_component
        final_scores.append(score)

    ranked = sorted(range(n_chunks), key=lambda x: final_scores[x], reverse=True)
    return ranked, final_scores, sub_scores


def _constrained_post_select(
    ranked_indices: List[int],
    sub_scores: List[List[float]],
    top_n: int,
) -> List[int]:
    """
    Enforcing constraint after chunk selection: per-subquery minimum coverage (>= tau_subquery)
    """
    if not ranked_indices or top_n <= 0:
        return []
    if not sub_scores:
        return ranked_indices[:top_n]

    m = len(sub_scores)
    tau_threshold = 0.52 if m >= 5 else 0.58

    def run_pass(
        tau_subquery: float,
    ) -> Optional[List[int]]:
        selected: List[int] = []
        selected_set = set()

        def can_add(idx: int) -> bool:
            return idx not in selected_set

        # Stage 1: satisfy each subquery with at least one strong chunk.
        for sub_i in range(m):
            if len(selected) >= top_n:
                break
            candidate = None
            for idx in ranked_indices:
                if can_add(idx) and sub_scores[sub_i][idx] >= tau_subquery:
                    candidate = idx
                    break
            if candidate is not None:
                selected.append(candidate)
                selected_set.add(candidate)

        # verify every subquery covered
        for sub_i in range(m):
            covered = any(sub_scores[sub_i][idx] >= tau_subquery for idx in selected)
            if not covered:
                return None

        # if there is space for more chunks
        while len(selected) < top_n:
            candidate = None

            for idx in ranked_indices:
                if not can_add(idx):
                    continue
                candidate = idx
                break

            if candidate is None:
                break

            selected.append(candidate)
            selected_set.add(candidate)

        sorted_selected = [idx for idx in ranked_indices if idx in selected_set][:top_n]
        return sorted_selected

    trials = [tau_threshold, tau_threshold - 0.05, tau_threshold - 0.08]

    for tau_trial in trials:
        selected = run_pass(tau_subquery=max(0.30, tau_trial))
        if selected:
            return selected

    # fallback: unconstrained reranked order
    return ranked_indices[:top_n]


def rerank_indices(
    query: str,
    chunks: List[str],
    mode: str,
    top_n: int,
) -> List[int]:
    """
    Ranking chunks
    """
    if mode == "cross_encoder":
        ranked = _rank_indices_with_cross_encoder(query, chunks)
    elif mode in {"multi_hop", "adaptive_multi_hop", "cross_encoder_multi_hop"}:
        if _is_multi_hop_query(query):
            ranked, final_scores, sub_scores = _rank_indices_with_subquery_coverage_details(query, chunks)
            ranked = _constrained_post_select(ranked_indices=ranked, sub_scores=sub_scores,top_n=top_n if top_n and top_n > 0 else len(chunks))
        else:
            ranked = _rank_indices_with_cross_encoder(query, chunks)
    else:
        ranked = list(range(len(chunks)))

    if top_n and top_n > 0 and mode in {"cross_encoder", "multi_hop", "adaptive_multi_hop", "cross_encoder_multi_hop"}:
        return ranked[:top_n]
    return ranked


# -------------------------- Reranking Router -----------------------------
def rerank(query: str, chunks: List[str], mode: str, top_n: int) -> List[str]:
    """
    Backwards-compatible
    """
    local_order = rerank_indices(query, chunks, mode=mode, top_n=top_n)
    return [chunks[i] for i in local_order]
