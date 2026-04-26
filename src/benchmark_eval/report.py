"""
src/benchmark_eval/report.py

Generates two kinds of markdown reports:

1. Single-run report  (generate_report)
   Comprehensive, verbose, with plain-English explanations, scores, tables,
   BLEU scores, and illustrative examples (best/worst answer, good/bad
   retrieval, faithful/unfaithful, rubric pass/fail).

2. AB-test comparative report  (generate_ab_report)
   Fixed-structure comparison table across all parameter combinations,
   with per-metric rankings and a plain-English recommendation.
"""

from __future__ import annotations

import json
import pathlib
import time
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pct(v: Optional[float], decimals: int = 1) -> str:
    if v is None:
        return "N/A"
    return f"{v * 100:.{decimals}f}%"


def _f(v: Optional[float], decimals: int = 2) -> str:
    if v is None:
        return "N/A"
    return f"{v:.{decimals}f}"


def _score_label(score: Optional[float]) -> str:
    """Plain-English label for -1/0/1 correctness scores."""
    if score is None:
        return "N/A"
    if score >= 1:
        return "✅ Fully correct"
    if score >= 0:
        return "⚠️ Partially correct"
    return "❌ Incorrect"


def _faith_label(verdict: Optional[str]) -> str:
    mapping = {
        "faithful":           "✅ Faithful",
        "partially_faithful": "⚠️ Partially faithful",
        "unfaithful":         "❌ Unfaithful",
        "uncertain":          "❓ Uncertain",
    }
    return mapping.get(verdict or "", "❓ Unknown")


def _bar(value: Optional[float], width: int = 20) -> str:
    """Simple ASCII progress bar."""
    if value is None:
        return "░" * width
    filled = round(value * width)
    return "█" * filled + "░" * (width - filled)


def _table(headers: List[str], rows: List[List[str]]) -> str:
    """Build a markdown table."""
    sep  = "| " + " | ".join("---" for _ in headers) + " |"
    head = "| " + " | ".join(headers) + " |"
    body = "\n".join(
        "| " + " | ".join(str(c) for c in row) + " |"
        for row in rows
    )
    return "\n".join([head, sep, body])


def _blockquote(text: str, max_chars: int = 0) -> str:
    """
    Wrap text in a markdown blockquote.
    Pass max_chars=0 (default) for no truncation.
    """
    truncated = (text[:max_chars] + "…") if max_chars and len(text) > max_chars else text
    lines = truncated.replace("\n", "  \n")
    return "> " + lines.replace("\n", "\n> ")


# ─────────────────────────────────────────────────────────────────────────────
# Example rendering helper
# ─────────────────────────────────────────────────────────────────────────────

def _render_example(
    lines: List[str],
    title: str,
    emoji: str,
    ex:    Optional[Dict],
) -> None:
    """Render one illustrative example block into the lines list."""
    if not ex:
        return

    W = lines.append

    W(f"### {emoji} {title}")
    W(f"")
    W(
        f"**Chapter {ex.get('chapter')} | "
        f"Difficulty: {ex.get('difficulty', '?').upper()} | "
        f"Record: `{ex.get('record_id', 'N/A')}`**"
    )
    W(f"")

    W(f"**Question:**")
    W(f"")
    W(_blockquote(ex.get("question", ""), 300))
    W(f"")

    # Full answer — no truncation
    W(f"**TokenSmith's Answer** *(full)*:")
    W(f"")
    W(_blockquote(ex.get("ts_answer", "")))
    W(f"")

    W(f"**Mock Answer** *(reference)*:")
    W(f"")
    W(_blockquote(ex.get("mock_answer", ""), 400))
    W(f"")

    # Per-rubric verdict breakdown
    rubric_results = ex.get("rubric_results", [])
    if rubric_results:
        W(f"**Rubric Breakdown:**")
        W(f"")
        for rr in rubric_results:
            v    = rr.get("verdict", "")
            icon = {"met": "✅", "partial": "⚠️", "not_met": "❌"}.get(v, "❓")
            W(f"- {icon} `{v.upper()}` — {rr.get('criterion', '')}")
            if rr.get("reason"):
                W(f"  - *{rr['reason']}*")
        W(f"")

    # Per-gold-chunk found/missed breakdown
    gold_status = ex.get("gold_chunk_status", [])
    if gold_status:
        W(f"**Gold Chunk Retrieval Breakdown:**")
        W(f"")
        for gs in gold_status:
            found     = gs.get("found", False)
            icon      = "✅" if found else "❌"
            label     = "RETRIEVED" if found else "MISSED"
            rank_note = (
                f" (found in rank {gs['found_in_rank']})"
                if gs.get("found_in_rank") else ""
            )
            W(f"- {icon} `{label}`{rank_note}")
            W(f"  - `{gs.get('gold_chunk', '')}`")
        W(f"")

    # Key metrics
    W(f"**Key Metrics:**")
    W(f"- Gold Chunk Coverage: {_pct(ex.get('gold_chunk_coverage_rate'))}")
    W(f"- Correctness (with ref): {_score_label(ex.get('correctness_score_with_ref'))}")
    W(f"- Rubric Met Rate: {_pct(ex.get('rubric_met_rate'))}")
    W(f"- Faithfulness: {_faith_label(ex.get('faithfulness_verdict'))}")
    bleu = ex.get("bleu_score")
    if bleu is not None:
        W(f"- BLEU Score vs Mock Answer: {bleu:.4f}")
    W(f"")

    explanation = ex.get("correctness_explanation_with_ref", "")
    if explanation:
        W(f"**Judge Explanation:**")
        W(f"")
        W(_blockquote(explanation, 400))
        W(f"")


# ─────────────────────────────────────────────────────────────────────────────
# Single-run report
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(
    run_label:    str,
    config_state: Dict,
    qac_file:     str,
    full_metrics: Dict,
    per_qac:      List[Dict],
    examples:     Dict,
    output_path:  pathlib.Path,
    judge_model:  str = "",
) -> None:
    """
    Generate a comprehensive, human-readable markdown report for one benchmark run.
    """
    overall = full_metrics.get("overall", {})
    by_diff = full_metrics.get("by_difficulty", {})
    by_chap = full_metrics.get("by_chapter", {})
    n       = overall.get("n", 0)

    lines: List[str] = []
    W = lines.append

    # ── Header ────────────────────────────────────────────────────────────────
    W(f"# TokenSmith Benchmark Report")
    W(f"")
    W(f"**Run label:** `{run_label}`  ")
    W(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}  ")
    W(f"**QAC file:** `{qac_file}`  ")
    W(f"**Questions evaluated:** {n}  ")
    judge_name = pathlib.Path(judge_model).name if judge_model else "N/A"
    W(f"**Judge model:** `{judge_name}`  ")
    W(f"")

    # ── Executive Summary ─────────────────────────────────────────────────────
    W(f"---")
    W(f"")
    W(f"## 📊 Executive Summary")
    W(f"")
    W(
        f"> This section gives a plain-English overview of how well TokenSmith "
        f"performed across all {n} benchmark questions."
    )
    W(f"")

    cov   = overall.get("gold_chunk_coverage_rate_mean")
    corr  = overall.get("correctness_score_with_ref_mean")
    rub      = overall.get("rubric_met_rate_mean_individual") or overall.get("rubric_met_rate_mean_all")
    rub_mode = "individual" if overall.get("rubric_met_rate_mean_individual") is not None else "all"
    faith = overall.get("faithfulness_faithful_rate") or overall.get("faithfulness_score_mean")
    prec  = overall.get("retrieval_precision_mean")
    bleu  = overall.get("bleu_score_mean")

    def _plain_coverage(v):
        if v is None:    return "could not be measured"
        if v >= 0.8:     return "**excellent** — almost all key information was retrieved"
        if v >= 0.6:     return "**good** — most key information was retrieved"
        if v >= 0.4:     return "**moderate** — about half the key information was retrieved"
        return "**poor** — less than half the key information was retrieved"

    def _plain_correctness(v):
        if v is None:    return "could not be measured"
        if v >= 0.8:     return "**excellent** — answers were largely correct"
        if v >= 0.4:     return "**moderate** — answers were partially correct on average"
        if v >= 0:       return "**mixed** — many answers were only partially correct"
        return "**poor** — answers were frequently incorrect"

    W(f"| Metric | Score | Plain English |")
    W(f"| --- | --- | --- |")
    W(f"| 🔍 Gold Chunk Coverage | {_pct(cov)} {_bar(cov, 10)} | Retrieval is {_plain_coverage(cov)} |")
    W(f"| ✅ Answer Correctness (with ref) | {_f(corr*100)}% | {_plain_correctness(corr)} |")
    W(f"| 📋 Rubric Satisfaction ({rub_mode} mode) | {_pct(rub)} | {_pct(rub)} of rubric criteria were fully met |")
    W(f"| 🎯 Retrieval Precision | {_pct(prec)} | {_pct(prec)} of retrieved chunks were useful |")
    W(f"| 🔒 Answer Faithfulness | {_pct(faith)} | How often answers stuck to retrieved info |")
    W(f"| 📝 BLEU vs Mock Answer | {_f(bleu, 4)} | N-gram overlap with the reference answer |")
    W(f"")

    err_pct = overall.get("error_rate_pct", 0)
    if err_pct > 0:
        W(f"> ⚠️ **{err_pct:.1f}% of questions produced errors** during execution.")
        W(f"")

    # ── Configuration ─────────────────────────────────────────────────────────
    W(f"---")
    W(f"")
    W(f"## ⚙️ Configuration")
    W(f"")
    W(f"The following TokenSmith settings were used for this benchmark run:")
    W(f"")
    W(f"```")
    for k, v in sorted(config_state.items()):
        W(f"  {k}: {v}")
    W(f"```")
    W(f"")

    # ── Retrieval Performance ─────────────────────────────────────────────────
    W(f"---")
    W(f"")
    W(f"## 🔍 Retrieval Performance")
    W(f"")
    W(
        f"> **What this measures:** For each question, the benchmark checks whether "
        f"the specific sentences needed to answer it were actually retrieved by TokenSmith. "
        f"A gold chunk is a verbatim sentence from the textbook that is necessary to answer "
        f"the question. If TokenSmith retrieves chunks that contain those sentences, "
        f"it had access to the right information."
    )
    W(f"")

    W(f"### Overall Retrieval Metrics")
    W(f"")
    perfect_pct = overall.get("perfect_gold_coverage_pct", 0)
    W(_table(
        ["Metric", "Value", "Meaning"],
        [
            ["Gold Chunk Coverage Rate", _pct(cov),
             "Fraction of required sentences found in retrieved chunks"],
            ["Perfect Coverage (100%)", f"{perfect_pct:.1f}%",
             "Questions where ALL gold chunks were retrieved"],
            ["Retrieval Precision", _pct(prec),
             "Fraction of retrieved chunks that contained at least one gold chunk"],
            ["Mean Retrieved Chunks",
             _f(overall.get("n_retrieved_chunks_mean")),
             "Average number of chunks TokenSmith retrieved per question"],
        ]
    ))
    W(f"")

    W(f"### Retrieval by Difficulty")
    W(f"")
    W(_table(
        ["Difficulty", "N", "Coverage Rate", "Perfect Coverage", "Precision"],
        [
            [
                diff.capitalize(),
                str(by_diff.get(diff, {}).get("n", 0)),
                _pct(by_diff.get(diff, {}).get("gold_chunk_coverage_rate_mean")),
                f"{by_diff.get(diff, {}).get('perfect_gold_coverage_pct', 0):.1f}%",
                _pct(by_diff.get(diff, {}).get("retrieval_precision_mean")),
            ]
            for diff in ("easy", "medium", "hard")
            if diff in by_diff
        ]
    ))
    W(f"")
    W(
        f"> **Note on difficulty:** Easy questions require 1-3 specific sentences. "
        f"Medium questions need 2-10 sentences, possibly spread across the chapter. "
        f"Hard questions require multiple concepts and reasoning, and typically have "
        f"more gold chunks spread further apart."
    )
    W(f"")

    # ── Answer Quality ────────────────────────────────────────────────────────
    W(f"---")
    W(f"")
    W(f"## ✅ Answer Quality")
    W(f"")
    W(
        f"> **What this measures:** Three complementary views of answer quality. "
        f"(1) Whether the answer satisfies the evaluation rubric criteria. "
        f"(2) An overall correctness score from -1 (wrong) to 1 (fully correct). "
        f"(3) Whether the answer was faithful to the retrieved chunks."
    )
    W(f"")

    W(f"### Rubric Satisfaction")
    W(f"")
    W(
        f"The rubric for each question specifies key points a correct answer must address."
    )
    W(f"")
    W(_table(
        ["Mode", "Met Rate", "Avg Score"],
        [
            [
                "Individual (one call per criterion)",
                _pct(overall.get("rubric_met_rate_mean_individual")),
                _f(overall.get("rubric_avg_score_individual")),
            ],
            [
                "All-at-once (one call for all criteria)",
                _pct(overall.get("rubric_met_rate_mean_all")),
                _f(overall.get("rubric_avg_score_all")),
            ],
        ]
    ))
    W(f"")

    W(f"### Rubric Satisfaction by Difficulty")
    W(f"")
    W(_table(
        ["Difficulty", "N", "Met Rate (Individual)", "Met Rate (All)"],
        [
            [
                diff.capitalize(),
                str(by_diff.get(diff, {}).get("n", 0)),
                _pct(by_diff.get(diff, {}).get("rubric_met_rate_mean_individual")),
                _pct(by_diff.get(diff, {}).get("rubric_met_rate_mean_all")),
            ]
            for diff in ("easy", "medium", "hard")
            if diff in by_diff
        ]
    ))
    W(f"")

    W(f"### Answer Correctness (-1 / 0 / 1)")
    W(f"")
    W(f"- **1** = fully correct, addresses all key rubric points")
    W(f"- **0** = partially correct, addresses some rubric points")
    W(f"- **-1** = incorrect or completely off-topic")
    W(f"")

    no_ref   = overall.get("correctness_no_ref_distribution",  {})
    with_ref = overall.get("correctness_with_ref_distribution", {})
    W(_table(
        ["Score", "Without Reference", "With Reference"],
        [
            [
                "✅ 1 (Fully correct)",
                f"{no_ref.get('1', 0)} ({_pct(no_ref.get('1', 0) / n if n else 0)})",
                f"{with_ref.get('1', 0)} ({_pct(with_ref.get('1', 0) / n if n else 0)})",
            ],
            [
                "⚠️ 0 (Partially correct)",
                f"{no_ref.get('0', 0)} ({_pct(no_ref.get('0', 0) / n if n else 0)})",
                f"{with_ref.get('0', 0)} ({_pct(with_ref.get('0', 0) / n if n else 0)})",
            ],
            [
                "❌ -1 (Incorrect)",
                f"{no_ref.get('-1', 0)} ({_pct(no_ref.get('-1', 0) / n if n else 0)})",
                f"{with_ref.get('-1', 0)} ({_pct(with_ref.get('-1', 0) / n if n else 0)})",
            ],
            [
                "Mean score",
                _f(overall.get("correctness_score_no_ref_mean")),
                _f(overall.get("correctness_score_with_ref_mean")),
            ],
        ]
    ))
    W(f"")

    # ── Rubric vs correctness examples ────────────────────────────────────────
    ex_match    = examples.get("rubric_high_correct")
    ex_mismatch = examples.get("rubric_high_mismatch")

    if ex_match or ex_mismatch:
        W(f"### Rubric Coverage vs Correctness Score Examples")
        W(f"")
        W(
            f"> These examples illustrate the relationship between rubric satisfaction "
            f"(did the answer address the key points?) and the judge's overall correctness "
            f"score. Sometimes an answer can tick many rubric boxes yet still be judged "
            f"incorrect overall — and vice versa."
        )
        W(f"")

    if ex_match:
        W(f"#### ✅ High Rubric Coverage + Correct Answer")
        W(f"")
        W(
            f"Rubric met rate: **{_pct(ex_match.get('rubric_met_rate'))}** | "
            f"Correctness: **{_score_label(ex_match.get('correctness_score_with_ref'))}**"
        )
        W(f"")
        W(f"**Question:** {ex_match.get('question', '')}")
        W(f"")
        W(f"**TokenSmith's Answer** *(full)*:")
        W(f"")
        W(_blockquote(ex_match.get("ts_answer", "")))
        W(f"")
        rubric_results = ex_match.get("rubric_results", [])
        if rubric_results:
            W(f"**Rubric Breakdown:**")
            W(f"")
            for rr in rubric_results:
                v    = rr.get("verdict", "")
                icon = {"met": "✅", "partial": "⚠️", "not_met": "❌"}.get(v, "❓")
                W(f"- {icon} `{v.upper()}` — {rr.get('criterion', '')}")
        W(f"")
        explanation = ex_match.get("correctness_explanation_with_ref", "")
        if explanation:
            W(f"**Judge Explanation:**")
            W(f"")
            W(_blockquote(explanation, 400))
            W(f"")

    if ex_mismatch:
        score = ex_mismatch.get("correctness_score_with_ref")
        score_word = "incorrect" if score == -1 else "partially correct"
        W(f"#### ⚠️ High Rubric Coverage but {score_word.title()} Answer")
        W(f"")
        W(
            f"Rubric met rate: **{_pct(ex_mismatch.get('rubric_met_rate'))}** | "
            f"Correctness: **{_score_label(score)}**  "
        )
        W(f"")
        W(
            f"> This case shows that satisfying rubric criteria does not guarantee "
            f"a fully correct answer — the judge found issues beyond what the rubric captured."
        )
        W(f"")
        W(f"**Question:** {ex_mismatch.get('question', '')}")
        W(f"")
        W(f"**TokenSmith's Answer** *(full)*:")
        W(f"")
        W(_blockquote(ex_mismatch.get("ts_answer", "")))
        W(f"")
        rubric_results = ex_mismatch.get("rubric_results", [])
        if rubric_results:
            W(f"**Rubric Breakdown:**")
            W(f"")
            for rr in rubric_results:
                v    = rr.get("verdict", "")
                icon = {"met": "✅", "partial": "⚠️", "not_met": "❌"}.get(v, "❓")
                W(f"- {icon} `{v.upper()}` — {rr.get('criterion', '')}")
        W(f"")
        explanation = ex_mismatch.get("correctness_explanation_with_ref", "")
        if explanation:
            W(f"**Judge Explanation:**")
            W(f"")
            W(_blockquote(explanation, 400))
            W(f"")

    # ── BLEU ──────────────────────────────────────────────────────────────────
    W(f"### BLEU Score (TokenSmith Answer vs Mock Answer)")
    W(f"")
    W(
        f"> BLEU measures n-gram overlap between TokenSmith's answer and the reference "
        f"mock answer. A score of 1.0 means identical, 0.0 means no overlap. In "
        f"open-ended QA, scores above 0.3 are generally considered good — the phrasing "
        f"does not need to match exactly."
    )
    W(f"")
    W(_table(
        ["Group", "N", "Mean BLEU", "Std Dev"],
        [
            [
                "Overall",
                str(n),
                _f(overall.get("bleu_score_mean"), 4),
                _f(overall.get("bleu_score_stdev"), 4),
            ]
        ] + [
            [
                diff.capitalize(),
                str(by_diff.get(diff, {}).get("n", 0)),
                _f(by_diff.get(diff, {}).get("bleu_score_mean"), 4),
                _f(by_diff.get(diff, {}).get("bleu_score_stdev"), 4),
            ]
            for diff in ("easy", "medium", "hard")
            if diff in by_diff
        ]
    ))
    W(f"")

    # ── Faithfulness ──────────────────────────────────────────────────────────
    W(f"### Answer Faithfulness")
    W(f"")
    W(
        f"> **What this measures:** Did the answer claim anything that was not supported "
        f"by the retrieved chunks? An unfaithful answer introduces facts or claims that "
        f"the system had no basis for — a sign of hallucination."
    )
    W(f"")
    faith_dist = overall.get("faithfulness_distribution", {})
    W(_table(
        ["Verdict", "Count", "Percentage"],
        [
            [
                _faith_label("faithful"),
                str(faith_dist.get("faithful", 0)),
                _pct(faith_dist.get("faithful", 0) / n if n else 0),
            ],
            [
                _faith_label("partially_faithful"),
                str(faith_dist.get("partially_faithful", 0)),
                _pct(faith_dist.get("partially_faithful", 0) / n if n else 0),
            ],
            [
                _faith_label("unfaithful"),
                str(faith_dist.get("unfaithful", 0)),
                _pct(faith_dist.get("unfaithful", 0) / n if n else 0),
            ],
            [
                _faith_label("uncertain"),
                str(faith_dist.get("uncertain", 0)),
                _pct(faith_dist.get("uncertain", 0) / n if n else 0),
            ],
        ]
    ))
    W(f"")

    # ── Answer Length ─────────────────────────────────────────────────────────
    W(f"### Answer Length")
    W(f"")
    W(_table(
        ["Metric", "Value"],
        [
            ["Mean word count",   _f(overall.get("answer_word_count_mean"), 1)],
            ["Std dev word count",_f(overall.get("answer_word_count_stdev"), 1)],
            ["Mean char count",   _f(overall.get("answer_char_count_mean"), 0)],
        ]
    ))
    W(f"")

    # ── Chunk Relevance ───────────────────────────────────────────────────────
    W(f"---")
    W(f"")
    W(f"## 🎯 Chunk Relevance")
    W(f"")
    W(
        f"> **What this measures:** Of the chunks TokenSmith retrieved, how many were "
        f"actually relevant to answering the question? Evaluated in two ways: "
        f"individually (one judge call per chunk) and in groups of 3."
    )
    W(f"")
    W(_table(
        ["Mode", "Relevance Rate"],
        [
            ["Individual (per chunk)",
             _pct(overall.get("chunk_relevance_rate_mean_individual"))],
            ["Group (per 3 chunks)",
             _pct(overall.get("chunk_relevance_rate_mean_group"))],
        ]
    ))
    W(f"")

    # ── Per-Chapter Breakdown ─────────────────────────────────────────────────
    if by_chap:
        W(f"---")
        W(f"")
        W(f"## 📚 Per-Chapter Breakdown")
        W(f"")
        chap_rows = []
        for chap_key, chap_data in sorted(by_chap.items()):
            chap_rows.append([
                chap_key.replace("_", " ").title(),
                str(chap_data.get("n", 0)),
                _pct(chap_data.get("gold_chunk_coverage_rate_mean")),
                _pct(chap_data.get("rubric_met_rate_mean_individual")),
                _f(chap_data.get("correctness_score_with_ref_mean")),
                _f(chap_data.get("bleu_score_mean"), 4),
                _f(chap_data.get("faithfulness_score_mean")),
            ])
        W(_table(
            ["Chapter", "N", "Gold Coverage", "Rubric Met",
             "Correctness", "BLEU", "Faithfulness"],
            chap_rows,
        ))
        W(f"")

    # ── Illustrative Examples ─────────────────────────────────────────────────
    W(f"---")
    W(f"")
    W(f"## 💡 Illustrative Examples")
    W(f"")
    W(
        f"> These examples are selected automatically to illustrate both strong and weak "
        f"performance. They are meant to give an intuitive feel for what the numbers mean."
    )
    W(f"")

    _render_example(lines, "Best Answer",  "🏆", examples.get("best_answer"))
    _render_example(lines, "Worst Answer", "🔻", examples.get("worst_answer"))

    # Retrieval examples
    W(f"### 🔍 Retrieval Examples")
    W(f"")
    W(
        f"> The following show a case where retrieval worked well and one where it did not."
    )
    W(f"")

    for title, emoji, key in [
        ("Best Retrieval — all gold chunks found", "✅", "best_retrieval"),
        ("Worst Retrieval — gold chunks missed",   "❌", "worst_retrieval"),
    ]:
        ex = examples.get(key)
        if not ex:
            continue
        W(f"#### {emoji} {title}")
        W(f"")
        W(f"**Question:** {ex.get('question', '')}")
        W(f"")
        W(f"**Gold chunks needed ({len(ex.get('gold_chunks', []))}):**")
        gold_status = ex.get("gold_chunk_status", [])
        for gs in gold_status:
            found     = gs.get("found", False)
            icon      = "✅" if found else "❌"
            rank_note = f" (rank {gs['found_in_rank']})" if gs.get("found_in_rank") else ""
            W(f"- {icon} `{gs.get('gold_chunk', '')}`{rank_note}")
        W(f"")
        W(f"**Coverage:** {_pct(ex.get('gold_chunk_coverage_rate'))}")
        W(f"")
        retrieved = ex.get("retrieved_chunks", [])
        if retrieved:
            W(f"**First retrieved chunk:**")
            W(f"")
            W(_blockquote(retrieved[0]))
            W(f"")

    # Faithfulness examples
    for title, emoji, key in [
        ("Faithful Answer Example",                          "✅", "faithful_example"),
        ("Unfaithful / Uncertain Answer Example",            "🚨", "unfaithful_example"),
    ]:
        ex = examples.get(key)
        if not ex:
            continue
        verdict = ex.get("faithfulness_verdict", "")
        W(f"### {emoji} {title}")
        W(f"")
        W(f"**Faithfulness verdict:** {_faith_label(verdict)}")
        W(f"")
        W(f"**Question:** {ex.get('question', '')}")
        W(f"")
        W(f"**TokenSmith's Answer** *(full)*:")
        W(f"")
        W(_blockquote(ex.get("ts_answer", "")))
        W(f"")
        explanation = ex.get("faithfulness_explanation", "")
        if explanation:
            W(f"**Judge Explanation:**")
            W(f"")
            W(_blockquote(explanation, 400))
            W(f"")
        unsupported = ex.get("unsupported_claims", [])
        if unsupported:
            W(f"**Unsupported claims identified by judge:**")
            for claim in unsupported:
                W(f"- {claim}")
            W(f"")

    # Rubric pass / fail examples
    _render_example(lines, "Rubric Pass Example", "✅", examples.get("rubric_pass_example"))
    _render_example(lines, "Rubric Fail Example", "❌", examples.get("rubric_fail_example"))

    # ── Footer ────────────────────────────────────────────────────────────────
    W(f"---")
    W(f"")
    W(
        f"*Report generated by TokenSmith Benchmark Evaluator on "
        f"{time.strftime('%Y-%m-%d %H:%M:%S')}*"
    )
    W(f"")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  [REPORT] Markdown report saved: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# AB-test comparative report
# ─────────────────────────────────────────────────────────────────────────────

def generate_ab_report(
    param_grid:          Dict[str, List],
    combination_results: List[Dict],   # [{label, params, full_metrics}, ...]
    output_path:         pathlib.Path,
) -> None:
    """
    Generate a comparative markdown report across all AB test combinations.

    combination_results entries:
        label        : str  (e.g. "top_k=5__rerank_mode=cross_encoder")
        params       : dict (the param values for this combination)
        full_metrics : dict (from build_full_metrics)
    """
    lines: List[str] = []
    W = lines.append

    W(f"# TokenSmith AB Test Comparative Report")
    W(f"")
    W(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}  ")
    W(f"**Combinations tested:** {len(combination_results)}  ")
    W(f"")

    W(f"## Parameters Varied")
    W(f"")
    for param, values in param_grid.items():
        W(f"- **`{param}`**: {values}")
    W(f"")

    # ── Main comparison table ─────────────────────────────────────────────────
    W(f"---")
    W(f"")
    W(f"## 📊 Results Comparison Table")
    W(f"")
    W(f"> ↑ = higher is better  |  * = context-dependent")
    W(f"")

    KEY_METRICS = [
        ("gold_chunk_coverage_rate_mean",         "Gold Coverage ↑"),
        ("retrieval_precision_mean",              "Retrieval Precision ↑"),
        ("rubric_met_rate_mean_individual",       "Rubric Met Rate ↑"),
        ("correctness_score_with_ref_mean",       "Correctness ↑"),
        ("faithfulness_score_mean",               "Faithfulness ↑"),
        ("chunk_relevance_rate_mean_individual",  "Chunk Relevance ↑"),
        ("bleu_score_mean",                       "BLEU ↑"),
        ("answer_word_count_mean",                "Avg Words *"),
    ]

    headers = ["Configuration"] + [m[1] for m in KEY_METRICS]
    rows = []
    for combo in combination_results:
        overall = combo["full_metrics"].get("overall", {})
        row = [f"`{combo['label']}`"]
        for key, _ in KEY_METRICS:
            val = overall.get(key)
            if key == "answer_word_count_mean":
                row.append(_f(val, 0) if val is not None else "N/A")
            elif key == "bleu_score_mean":
                row.append(_f(val, 4) if val is not None else "N/A")
            else:
                row.append(_pct(val) if val is not None else "N/A")
        rows.append(row)
    W(_table(headers, rows))
    W(f"")

    # ── Per-metric rankings ───────────────────────────────────────────────────
    W(f"---")
    W(f"")
    W(f"## 🏅 Per-Metric Rankings")
    W(f"")

    for key, label in KEY_METRICS:
        if key == "answer_word_count_mean":
            continue
        W(f"### {label}")
        W(f"")
        scored = [
            (combo["label"], combo["full_metrics"].get("overall", {}).get(key))
            for combo in combination_results
        ]
        scored_valid = [(lbl, v) for lbl, v in scored if v is not None]
        scored_valid.sort(key=lambda x: x[1], reverse=True)
        W(_table(
            ["Rank", "Configuration", label],
            [
                [f"#{i+1}", f"`{lbl}`",
                 _f(val, 4) if key == "bleu_score_mean" else _pct(val)]
                for i, (lbl, val) in enumerate(scored_valid)
            ]
        ))
        W(f"")

    # ── Winner analysis ───────────────────────────────────────────────────────
    W(f"---")
    W(f"")
    W(f"## 🥇 Winner Analysis")
    W(f"")
    W(f"> Which configuration wins on the most metrics?")
    W(f"")

    win_counts: Dict[str, int] = {combo["label"]: 0 for combo in combination_results}
    for key, _ in KEY_METRICS:
        if key == "answer_word_count_mean":
            continue
        scored = [
            (combo["label"], combo["full_metrics"].get("overall", {}).get(key))
            for combo in combination_results
        ]
        scored_valid = [(lbl, v) for lbl, v in scored if v is not None]
        if scored_valid:
            winner = max(scored_valid, key=lambda x: x[1])[0]
            win_counts[winner] = win_counts.get(winner, 0) + 1

    win_rows = sorted(win_counts.items(), key=lambda x: x[1], reverse=True)
    W(_table(
        ["Configuration", "Metrics Won"],
        [[f"`{lbl}`", str(cnt)] for lbl, cnt in win_rows],
    ))
    W(f"")

    overall_winner = win_rows[0][0] if win_rows else "N/A"
    W(f"### 💡 Recommendation")
    W(f"")
    W(f"Based on the number of metrics won, the recommended configuration is:")
    W(f"")
    W(f"> **`{overall_winner}`**")
    W(f"")
    W(
        f"However, review the full table above to check whether this configuration "
        f"has any significant weaknesses on metrics that matter most for your use case."
    )
    W(f"")

    # ── Per-difficulty comparison ─────────────────────────────────────────────
    W(f"---")
    W(f"")
    W(f"## 📈 Performance by Difficulty")
    W(f"")
    for diff in ("easy", "medium", "hard"):
        W(f"### {diff.capitalize()} Questions")
        W(f"")
        diff_rows = []
        for combo in combination_results:
            diff_data = combo["full_metrics"].get("by_difficulty", {}).get(diff, {})
            if not diff_data:
                continue
            diff_rows.append([
                f"`{combo['label']}`",
                str(diff_data.get("n", 0)),
                _pct(diff_data.get("gold_chunk_coverage_rate_mean")),
                _pct(diff_data.get("rubric_met_rate_mean_individual")),
                _f(diff_data.get("correctness_score_with_ref_mean")),
                _f(diff_data.get("bleu_score_mean"), 4),
                _f(diff_data.get("faithfulness_score_mean")),
            ])
        if diff_rows:
            W(_table(
                ["Configuration", "N", "Gold Coverage", "Rubric Met",
                 "Correctness", "BLEU", "Faithfulness"],
                diff_rows,
            ))
        W(f"")

    W(f"---")
    W(f"")
    W(
        f"*AB Test report generated by TokenSmith Benchmark Evaluator on "
        f"{time.strftime('%Y-%m-%d %H:%M:%S')}*"
    )
    W(f"")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  [REPORT] AB test comparative report saved: {output_path}")