"""
src/benchmark_eval/metrics.py

Deterministic metrics computed from benchmark results and judge outputs.
No LLM calls here — pure computation over the collected data.

Metrics computed at three levels of granularity
------------------------------------------------
1. Per-QAC   — one row per question, all metrics as fields
2. Per-group — aggregated by difficulty (easy / medium / hard)
3. Overall   — across all questions

Metrics catalogue
-----------------
Retrieval:
  gold_chunk_coverage_rate  : fraction of gold chunks found in retrieved chunks
  n_gold_chunks             : how many gold chunks the QAC has
  n_gold_chunks_covered     : how many were found
  retrieval_precision       : fraction of retrieved chunks containing any gold chunk
  n_retrieved_chunks        : number of chunks TokenSmith retrieved

Answer:
  answer_word_count         : word count of TokenSmith's answer
  answer_char_count         : char count of TokenSmith's answer

Judge — chunk relevance (individual mode):
  chunk_relevance_rate      : fraction of retrieved chunks judged relevant

Judge — chunk relevance (group mode):
  chunk_relevance_rate_group

Judge — rubric satisfaction (individual mode):
  rubric_met_rate           : fraction of rubric criteria fully met
  rubric_partial_rate       : fraction partially met

Judge — rubric satisfaction (all mode):
  rubric_met_rate_all
  rubric_partial_rate_all

Judge — answer correctness:
  correctness_score_no_ref  : -1/0/1 without reference
  correctness_score_with_ref: -1/0/1 with reference

Judge — faithfulness:
  faithfulness_verdict      : faithful / partially_faithful / unfaithful / uncertain
  faithfulness_score        : 1 / 0.5 / 0 / None (numeric form for aggregation)

Error:
  had_error                 : whether get_answer() raised during this QAC
"""

from __future__ import annotations

import json
import pathlib
import statistics
from typing import Any, Dict, List, Optional
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


# ─────────────────────────────────────────────────────────────────────────────
# Per-QAC metric extraction
# ─────────────────────────────────────────────────────────────────────────────

_FAITHFULNESS_SCORE = {
    "faithful":           1.0,
    "partially_faithful": 0.5,
    "unfaithful":         0.0,
    "uncertain":          None,
}

_VERDICT_SCORE = {
    "relevant":     1,
    "not_relevant": 0,
    "uncertain":    None,
}

_RUBRIC_SCORE = {
    "met":     1.0,
    "partial": 0.5,
    "not_met": 0.0,
}


def extract_per_qac_metrics(result: Dict, judgements: Dict) -> Dict:
    """
    Extract all metrics for one QAC result + its judge outputs.
    Returns a flat dict suitable for a CSV row or JSON record.
    """
    qac              = result["qac"]
    ts_answer        = result.get("ts_answer", "")
    retrieved_chunks = result.get("retrieved_chunks", [])
    error            = result.get("error")

    # ── Identity ──────────────────────────────────────────────────────────────
    m: Dict[str, Any] = {
        "record_id":    qac.get("record_id", ""),
        "chapter":      qac.get("chapter"),
        "window_pages": qac.get("window_pages"),
        "difficulty":   qac.get("difficulty", "?"),
        "question":     qac.get("question", ""),
        "had_error":    bool(error),
    }

    # ── Answer size ───────────────────────────────────────────────────────────
    m["answer_word_count"] = len(ts_answer.split())
    m["answer_char_count"] = len(ts_answer)
    m["n_retrieved_chunks"] = len(retrieved_chunks)

    # ── Gold chunk presence (judge eval 4 — deterministic) ────────────────────
    gcp = judgements.get("gold_chunk_presence", {})
    m["n_gold_chunks"]         = len(qac.get("gold_chunks", []))
    m["n_gold_chunks_covered"] = len(gcp.get("covered_gold_chunks", []))
    m["gold_chunk_coverage_rate"] = gcp.get("coverage_rate", 0.0)
    m["retrieval_precision"]   = gcp.get("retrieval_precision", 0.0)

    # ── Chunk relevance — individual ──────────────────────────────────────────
    ind_rel = judgements.get("chunk_relevance", {}).get("individual", [])
    if ind_rel:
        scores = [_VERDICT_SCORE.get(r["verdict"]) for r in ind_rel]
        valid  = [s for s in scores if s is not None]
        m["chunk_relevance_rate"] = sum(valid) / len(valid) if valid else None
        m["n_chunks_relevant_individual"] = sum(1 for r in ind_rel if r["verdict"] == "relevant")
    else:
        m["chunk_relevance_rate"] = None
        m["n_chunks_relevant_individual"] = None

    # ── Chunk relevance — group ───────────────────────────────────────────────
    grp_rel = judgements.get("chunk_relevance", {}).get("group", [])
    if grp_rel:
        scores = [_VERDICT_SCORE.get(r["verdict"]) for r in grp_rel]
        valid  = [s for s in scores if s is not None]
        m["chunk_relevance_rate_group"] = sum(valid) / len(valid) if valid else None
    else:
        m["chunk_relevance_rate_group"] = None

    # ── Rubric satisfaction — individual ──────────────────────────────────────
    # Use whichever mode ran — individual takes priority, fall back to all
    ind_rub = (
        judgements.get("rubric_satisfaction", {}).get("individual")
        or judgements.get("rubric_satisfaction", {}).get("all")
        or []
    )
    if ind_rub:
        scores = [_RUBRIC_SCORE.get(r["verdict"], 0.0) for r in ind_rub]
        m["rubric_met_rate"]             = sum(1 for r in ind_rub if r["verdict"] == "met") / len(ind_rub)
        m["rubric_partial_rate"]         = sum(1 for r in ind_rub if r["verdict"] == "partial") / len(ind_rub)
        m["rubric_avg_score_individual"] = sum(scores) / len(scores)
        m["n_rubric_criteria"]           = len(ind_rub)
    else:
        m["rubric_met_rate"]             = None
        m["rubric_partial_rate"]         = None
        m["rubric_avg_score_individual"] = None
        m["n_rubric_criteria"]           = 0

    # ── Rubric satisfaction — all mode ────────────────────────────────────────
    all_rub = (
        judgements.get("rubric_satisfaction", {}).get("all")
        or judgements.get("rubric_satisfaction", {}).get("individual")
        or []
    )
    if all_rub:
        scores  = [_RUBRIC_SCORE.get(r["verdict"], 0.0) for r in all_rub]
        m["rubric_met_rate_all"]     = sum(1 for r in all_rub if r["verdict"] == "met") / len(all_rub)
        m["rubric_partial_rate_all"] = sum(1 for r in all_rub if r["verdict"] == "partial") / len(all_rub)
        m["rubric_avg_score_all"]    = sum(scores) / len(scores)
    else:
        m["rubric_met_rate_all"]     = None
        m["rubric_partial_rate_all"] = None
        m["rubric_avg_score_all"]    = None

    # ── Answer correctness ────────────────────────────────────────────────────
    corr = judgements.get("answer_correctness", {})
    m["correctness_score_no_ref"]   = corr.get("without_reference", {}).get("score")
    m["correctness_score_with_ref"] = corr.get("with_reference", {}).get("score")

    # ── Faithfulness ──────────────────────────────────────────────────────────
    faith = judgements.get("faithfulness", {})
    verdict = faith.get("verdict", "uncertain")
    m["faithfulness_verdict"] = verdict
    m["faithfulness_score"]   = _FAITHFULNESS_SCORE.get(verdict)

    # ── BLEU score (mock answer vs TokenSmith answer) ─────────────────────────
    mock_answer = qac.get("mock_answer", "")
    if mock_answer.strip() and ts_answer.strip():
        try:
            reference  = [mock_answer.lower().split()]
            hypothesis = ts_answer.lower().split()
            smoother   = SmoothingFunction().method1
            m["bleu_score"] = sentence_bleu(reference, hypothesis,
                                            smoothing_function=smoother)
        except Exception:
            m["bleu_score"] = None
    else:
        m["bleu_score"] = None

    return m


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_mean(values: List) -> Optional[float]:
    valid = [v for v in values if v is not None]
    return statistics.mean(valid) if valid else None


def _safe_stdev(values: List) -> Optional[float]:
    valid = [v for v in values if v is not None]
    return statistics.stdev(valid) if len(valid) > 1 else None


def _pct(numerator: int, denominator: int) -> float:
    return (numerator / denominator * 100) if denominator else 0.0


def _distribution(values: List, bins: List) -> Dict[str, int]:
    """Count how many values fall into each string category."""
    counts: Dict[str, int] = {b: 0 for b in bins}
    for v in values:
        key = str(v) if v is not None else "unknown"
        if key in counts:
            counts[key] += 1
        else:
            counts["unknown"] = counts.get("unknown", 0) + 1
    return counts


def aggregate_metrics(per_qac: List[Dict], group_name: str = "all") -> Dict:
    """
    Aggregate per-QAC metrics into summary statistics for a group of results.
    group_name is a label (e.g. "all", "easy", "chapter_3").
    """
    n = len(per_qac)
    if n == 0:
        return {"group": group_name, "n": 0}

    def col(key: str) -> List:
        return [r.get(key) for r in per_qac]

    agg: Dict[str, Any] = {
        "group": group_name,
        "n":     n,
    }

    # Error rate
    errors = sum(1 for r in per_qac if r.get("had_error"))
    agg["error_rate_pct"] = _pct(errors, n)

    # Answer size
    agg["answer_word_count_mean"]  = _safe_mean(col("answer_word_count"))
    agg["answer_word_count_stdev"] = _safe_stdev(col("answer_word_count"))
    agg["answer_char_count_mean"]  = _safe_mean(col("answer_char_count"))

    # Retrieval
    agg["n_retrieved_chunks_mean"]      = _safe_mean(col("n_retrieved_chunks"))
    agg["gold_chunk_coverage_rate_mean"]= _safe_mean(col("gold_chunk_coverage_rate"))
    agg["gold_chunk_coverage_rate_stdev"]= _safe_stdev(col("gold_chunk_coverage_rate"))
    agg["retrieval_precision_mean"]     = _safe_mean(col("retrieval_precision"))

    # Perfect coverage rate (all gold chunks found)
    perfect = sum(
        1 for r in per_qac
        if r.get("n_gold_chunks", 0) > 0
        and r.get("n_gold_chunks_covered") == r.get("n_gold_chunks")
    )
    agg["perfect_gold_coverage_pct"] = _pct(perfect, n)

    # Chunk relevance
    agg["chunk_relevance_rate_mean_individual"] = _safe_mean(col("chunk_relevance_rate"))
    agg["chunk_relevance_rate_mean_group"]      = _safe_mean(col("chunk_relevance_rate_group"))

    # Rubric satisfaction
    agg["rubric_met_rate_mean_individual"] = _safe_mean(col("rubric_met_rate"))
    agg["rubric_met_rate_mean_all"]        = _safe_mean(col("rubric_met_rate_all"))
    agg["rubric_avg_score_individual"]     = _safe_mean(col("rubric_avg_score_individual"))
    agg["rubric_avg_score_all"]            = _safe_mean(col("rubric_avg_score_all"))

    # Answer correctness
    agg["correctness_score_no_ref_mean"]   = _safe_mean(col("correctness_score_no_ref"))
    agg["correctness_score_with_ref_mean"] = _safe_mean(col("correctness_score_with_ref"))
    agg["correctness_no_ref_distribution"] = _distribution(
        col("correctness_score_no_ref"), ["-1", "0", "1"]
    )
    agg["correctness_with_ref_distribution"] = _distribution(
        col("correctness_score_with_ref"), ["-1", "0", "1"]
    )

    # Faithfulness
    agg["faithfulness_score_mean"] = _safe_mean(col("faithfulness_score"))
    agg["faithfulness_distribution"] = _distribution(
        col("faithfulness_verdict"),
        ["faithful", "partially_faithful", "unfaithful", "uncertain"],
    )
    agg["bleu_score_mean"]  = _safe_mean(col("bleu_score"))
    agg["bleu_score_stdev"] = _safe_stdev(col("bleu_score"))

    return agg


def build_full_metrics(per_qac: List[Dict]) -> Dict:
    """
    Build aggregated metrics at all three levels:
      overall → by difficulty → by chapter
    """
    overall = aggregate_metrics(per_qac, group_name="overall")

    by_difficulty: Dict[str, Dict] = {}
    for diff in ("easy", "medium", "hard"):
        subset = [r for r in per_qac if r.get("difficulty") == diff]
        if subset:
            by_difficulty[diff] = aggregate_metrics(subset, group_name=diff)

    chapters = sorted({r.get("chapter") for r in per_qac if r.get("chapter") is not None})
    by_chapter: Dict[str, Dict] = {}
    for chap in chapters:
        subset = [r for r in per_qac if r.get("chapter") == chap]
        by_chapter[f"chapter_{chap}"] = aggregate_metrics(
            subset, group_name=f"chapter_{chap}"
        )

    return {
        "overall":      overall,
        "by_difficulty": by_difficulty,
        "by_chapter":   by_chapter,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Persistence helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_metrics_csv(per_qac: List[Dict], path: pathlib.Path) -> None:
    """Write per-QAC metrics to a CSV file."""
    import csv
    if not per_qac:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(per_qac[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(per_qac)
    print(f"  [METRICS] CSV saved: {path}")


def save_metrics_json(full_metrics: Dict, path: pathlib.Path) -> None:
    """Write aggregated metrics summary to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(full_metrics, f, indent=2, ensure_ascii=False)
    print(f"  [METRICS] JSON saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Example selection helpers (used by report generator)
# ─────────────────────────────────────────────────────────────────────────────

def find_examples(
    per_qac:    List[Dict],
    results:    List[Dict],
    judgements: List[Dict],
) -> Dict:
    """
    Find illustrative examples for the report:
      best_answer           — highest correctness score (with ref)
      worst_answer          — lowest correctness score (with ref)
      best_retrieval        — highest gold chunk coverage
      worst_retrieval       — lowest gold chunk coverage
      faithful_example      — answer judged faithful
      unfaithful_example    — unfaithful first, then partially_faithful, then uncertain
      rubric_pass_example   — all rubric criteria met
      rubric_fail_example   — most rubric criteria failed
    """
    paired = list(zip(per_qac, results, judgements))
    if not paired:
        return {}

    def _best(key, reverse=True):
        valid = [(m, r, j) for m, r, j in paired if m.get(key) is not None]
        if not valid:
            return None
        return sorted(valid, key=lambda x: x[0][key], reverse=reverse)[0]

    examples: Dict[str, Any] = {}

    b = _best("correctness_score_with_ref", reverse=True)
    if b:
        examples["best_answer"] = _summarise_example(*b)

    w = _best("correctness_score_with_ref", reverse=False)
    if w:
        examples["worst_answer"] = _summarise_example(*w)

    br = _best("gold_chunk_coverage_rate", reverse=True)
    if br:
        examples["best_retrieval"] = _summarise_example(*br)

    wr = _best("gold_chunk_coverage_rate", reverse=False)
    if wr:
        examples["worst_retrieval"] = _summarise_example(*wr)

    # Faithful example
    faithful_match = next(
        ((m, r, j) for m, r, j in paired if m.get("faithfulness_verdict") == "faithful"),
        None,
    )
    if faithful_match:
        examples["faithful_example"] = _summarise_example(*faithful_match)

    # Unfaithful example — try unfaithful first, then partially_faithful, then uncertain
    unfaithful_match = (
        next(
            ((m, r, j) for m, r, j in paired
             if m.get("faithfulness_verdict") == "unfaithful"),
            None,
        )
        or next(
            ((m, r, j) for m, r, j in paired
             if m.get("faithfulness_verdict") == "partially_faithful"),
            None,
        )
        or next(
            ((m, r, j) for m, r, j in paired
             if m.get("faithfulness_verdict") == "uncertain"),
            None,
        )
    )
    if unfaithful_match:
        examples["unfaithful_example"] = _summarise_example(*unfaithful_match)

    # Rubric examples
    rubric_pairs = sorted(
        [(m, r, j) for m, r, j in paired if m.get("rubric_met_rate") is not None],
        key=lambda x: x[0]["rubric_met_rate"],
        reverse=True,
    )
    if rubric_pairs:
        examples["rubric_pass_example"] = _summarise_example(*rubric_pairs[0])
    if len(rubric_pairs) > 1:
        examples["rubric_fail_example"] = _summarise_example(*rubric_pairs[-1])

    # Rubric high + correctness mismatch
    # Find cases where rubric coverage is above 80% but model scored wrong
    rubric_high = [
        (m, r, j) for m, r, j in paired
        if (m.get("rubric_met_rate") or 0) >= 0.8
        and m.get("correctness_score_with_ref") is not None
    ]

    # Positive case: rubric high AND correctness == 1
    rubric_high_correct = next(
        ((m, r, j) for m, r, j in rubric_high
         if m.get("correctness_score_with_ref") == 1),
        None,
    )
    if rubric_high_correct:
        examples["rubric_high_correct"] = _summarise_example(*rubric_high_correct)

    # Mismatch case: rubric high but correctness wrong (-1 first, then 0)
    rubric_high_mismatch = (
        next(
            ((m, r, j) for m, r, j in rubric_high
             if m.get("correctness_score_with_ref") == -1),
            None,
        )
        or next(
            ((m, r, j) for m, r, j in rubric_high
             if m.get("correctness_score_with_ref") == 0),
            None,
        )
    )
    if rubric_high_mismatch:
        examples["rubric_high_mismatch"] = _summarise_example(*rubric_high_mismatch)

    return examples


def _summarise_example(metrics: Dict, result: Dict, judgement: Dict) -> Dict:
    qac = result.get("qac", {})

    # Per-rubric verdict from whichever mode ran
    rubric_results = (
        judgement.get("rubric_satisfaction", {}).get("all")
        or judgement.get("rubric_satisfaction", {}).get("individual")
        or []
    )

    # Per-gold-chunk found status
    gold_chunk_status = judgement.get("gold_chunk_presence", {}).get("per_gold_chunk", [])

    return {
        "record_id":              metrics.get("record_id"),
        "difficulty":             metrics.get("difficulty"),
        "chapter":                metrics.get("chapter"),
        "question":               qac.get("question", ""),
        "ts_answer":              result.get("ts_answer", ""),   # full, no truncation
        "mock_answer":            qac.get("mock_answer", ""),
        "gold_chunks":            qac.get("gold_chunks", []),
        "gold_chunk_status":      gold_chunk_status,             # [{gold_chunk, found, found_in_rank}]
        "rubric":                 qac.get("rubric", []),
        "rubric_results":         rubric_results,                # [{criterion, verdict, reason}]
        "retrieved_chunks":       [c.get("content", "") for c in result.get("retrieved_chunks", [])],
        "gold_chunk_coverage_rate": metrics.get("gold_chunk_coverage_rate"),
        "correctness_score_no_ref": metrics.get("correctness_score_no_ref"),
        "correctness_score_with_ref": metrics.get("correctness_score_with_ref"),
        "rubric_met_rate":        metrics.get("rubric_met_rate"),
        "faithfulness_verdict":   metrics.get("faithfulness_verdict"),
        "bleu_score":             metrics.get("bleu_score"),
        "correctness_explanation_no_ref": (
            judgement.get("answer_correctness", {})
                     .get("without_reference", {}).get("explanation", "")
        ),
        "correctness_explanation_with_ref": (
            judgement.get("answer_correctness", {})
                     .get("with_reference", {}).get("explanation", "")
        ),
        "faithfulness_explanation": judgement.get("faithfulness", {}).get("explanation", ""),
        "unsupported_claims":     judgement.get("faithfulness", {}).get("unsupported_claims", []),
    }