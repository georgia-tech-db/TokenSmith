#!/usr/bin/env python3
"""
src/benchmark_eval/run_external_benchmark.py

Run an external benchmark JSON file through TokenSmith and the LLM judge pipeline.

The external benchmark format:
    [
      {
        "id": "2.10",
        "question": "...",
        "answer": "...",          ← used as mock reference answer
        "must_rubric": [...],     ← required criteria (counted in overall score)
        "optional_rubric": [...]  ← bonus criteria (scored but not counted overall)
      },
      ...
    ]

Usage
-----
    # Run with local judge (default)
    python3 src/benchmark_eval/run_external_benchmark.py \\
        --benchmark data/my_benchmark.json \\
        --label my_run

    # Run with OpenRouter judge
    python3 src/benchmark_eval/run_external_benchmark.py \\
        --benchmark data/my_benchmark.json \\
        --label my_run_72b \\
        --judge-backend openrouter \\
        --judge-model qwen/qwen-2.5-72b-instruct

    # Skip judge (retrieval only, much faster)
    python3 src/benchmark_eval/run_external_benchmark.py \\
        --benchmark data/my_benchmark.json \\
        --label my_run_fast \\
        --no-judge

    # Dry run
    python3 src/benchmark_eval/run_external_benchmark.py \\
        --benchmark data/my_benchmark.json \\
        --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys
import textwrap
import time
from argparse import Namespace
from copy import deepcopy
from typing import Optional

# ── Project root on path ──────────────────────────────────────────────────────
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Load .env from project root if present
_env_path = _PROJECT_ROOT / ".env"
if _env_path.exists():
    with open(_env_path) as _ef:
        for _line in _ef:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                _v = _v.strip().strip('"').strip("'")
                os.environ.setdefault(_k.strip(), _v.strip())

import yaml
from src.config import RAGConfig
from src.instrumentation.logging import get_logger
from src.main import get_answer
from src.ranking.ranker import EnsembleRanker
from src.retriever import (
    BM25Retriever, FAISSRetriever, IndexKeywordRetriever, load_artifacts,
)
from src.benchmark_eval.judge_client import JudgeClient, safe_verdict

DEFAULT_CONFIG   = "config/config.yaml"
DEFAULT_OUT_ROOT = "benchmark_results"
INDEX_PREFIX     = "textbook_index"


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark loading
# ─────────────────────────────────────────────────────────────────────────────

def load_benchmark(path: pathlib.Path) -> list[dict]:
    """Load and validate the external benchmark JSON file."""
    if not path.exists():
        print(f"ERROR: Benchmark file not found: {path}", file=sys.stderr)
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        print("ERROR: Benchmark file must be a JSON array.", file=sys.stderr)
        sys.exit(1)
    valid = []
    for i, item in enumerate(data):
        if "question" not in item:
            print(f"  [WARN] Item {i} missing 'question' — skipping")
            continue
        if "id" not in item:
            item["id"] = str(i)
        item.setdefault("answer", "")
        item.setdefault("must_rubric", [])
        item.setdefault("optional_rubric", [])
        valid.append(item)
    print(f"  Loaded {len(valid)} benchmark questions from {path.name}")
    return valid


def load_previous_ts_results(run_label: str) -> dict[str, dict]:
    """
    Load ts_answer and retrieved_chunks from a previous run's full_results.json.
    Returns a dict keyed by question id.
    """
    path = pathlib.Path(DEFAULT_OUT_ROOT) / run_label / "full_results.json"
    if not path.exists():
        print(f"ERROR: Previous run not found: {path}", file=sys.stderr)
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    index = {r["id"]: r for r in records}
    print(f"  Loaded {len(index)} previous results from {path}")
    return index

# ─────────────────────────────────────────────────────────────────────────────
# TokenSmith interface
# ─────────────────────────────────────────────────────────────────────────────

def load_ts_artifacts(cfg: RAGConfig) -> dict:
    try:
        artifacts_dir = cfg.get_artifacts_directory()
        faiss_idx, bm25_idx, chunks, sources, meta = load_artifacts(
            artifacts_dir, INDEX_PREFIX
        )
        retrievers = [FAISSRetriever(faiss_idx, cfg.embed_model), BM25Retriever(bm25_idx)]
        if cfg.ranker_weights.get("index_keywords", 0) > 0:
            retrievers.append(
                IndexKeywordRetriever(cfg.extracted_index_path, cfg.page_to_chunk_map_path)
            )
        ranker = EnsembleRanker(
            ensemble_method=cfg.ensemble_method,
            weights=cfg.ranker_weights,
            rrf_k=int(cfg.rrf_k),
        )
        return {"chunks": chunks, "sources": sources, "meta": meta,
                "retrievers": retrievers, "ranker": ranker}
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load TokenSmith artifacts: {exc}\n"
            "Run 'python3 main.py index' first."
        ) from exc


def run_question_through_ts(
    question: str, cfg: RAGConfig, artifacts: dict, logger
) -> tuple[str, list[dict], Optional[str]]:
    """
    Run one question through TokenSmith in test mode.
    Returns (ts_answer, chunks_info, hyde_query).
    """
    cfg_copy = deepcopy(cfg)
    cfg_copy.enable_history = False

    args = Namespace(system_prompt_mode="", double_prompt=False)
    try:
        result = get_answer(
            question=question,
            cfg=cfg_copy,
            args=args,
            logger=logger,
            console=None,
            artifacts=artifacts,
            is_test_mode=True,
        )
        if isinstance(result, tuple):
            ts_answer, chunks_info, hyde_query = result
        else:
            ts_answer, chunks_info, hyde_query = result, [], None
        return ts_answer or "", chunks_info or [], hyde_query
    except Exception as exc:
        import traceback
        print(f"    [ERROR] TokenSmith failed: {exc}")
        traceback.print_exc()
        return "", [], None


# ─────────────────────────────────────────────────────────────────────────────
# Judge prompts
# ─────────────────────────────────────────────────────────────────────────────

def _rubric_individual_prompt(question: str, answer: str, criterion: str,
                               is_optional: bool) -> str:
    optional_note = (
        "\nNote: this is an OPTIONAL criterion. It is a bonus point "
        "if addressed but is NOT required for a fully correct answer.\n"
        if is_optional else ""
    )
    return textwrap.dedent(f"""
        You are evaluating whether a student answer satisfies a rubric criterion.
        {optional_note}
        QUESTION:
        {question}

        STUDENT ANSWER:
        {answer}

        RUBRIC CRITERION:
        {criterion}

        Does the answer satisfy this criterion?

        Respond with ONLY this JSON object:
        {{
          "verdict": "met" | "not_met" | "partial",
          "reason": "<one sentence explaining your decision>"
        }}
    """).strip()


def _rubric_all_prompt(question: str, answer: str,
                        criteria: list[str], label: str) -> str:
    crit_block = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(criteria))
    crit_keys  = ", ".join(
        f'"criterion_{i+1}": {{"verdict": "met|not_met|partial", "reason": "..."}}'
        for i in range(len(criteria))
    )
    return textwrap.dedent(f"""
        You are evaluating whether a student answer satisfies rubric criteria ({label}).

        QUESTION:
        {question}

        STUDENT ANSWER:
        {answer}

        RUBRIC CRITERIA:
        {crit_block}

        For each criterion, decide: met | not_met | partial

        Respond with ONLY this JSON object:
        {{
          {crit_keys}
        }}
    """).strip()


def _correctness_prompt(question: str, answer: str, must_rubric: list[str],
                         mock_answer: Optional[str] = None) -> str:
    rubric_block = "\n".join(f"  - {r}" for r in must_rubric)
    ref_section  = (
        f"\nREFERENCE ANSWER (other phrasings can also be correct):\n{mock_answer}\n"
        if mock_answer else ""
    )
    return textwrap.dedent(f"""
        You are scoring the quality of a student answer to a question.

        QUESTION:
        {question}

        STUDENT ANSWER:
        {answer}
        {ref_section}
        KEY RUBRIC POINTS a correct answer must address:
        {rubric_block}

        Score the answer:
          1  = fully correct — addresses all key rubric points accurately
          0  = partially correct — addresses some rubric points but misses others
         -1  = incorrect or completely off-topic

        Respond with ONLY this JSON object:
        {{
          "score": 1 | 0 | -1,
          "explanation": "<2-3 sentences explaining the score>",
          "strengths": ["<strength 1>"],
          "weaknesses": ["<weakness 1>"]
        }}
    """).strip()


def _chunk_relevance_group_prompt(question: str, chunks: list[str]) -> str:
    chunk_block = "\n\n".join(f"CHUNK {i+1}:\n{c}" for i, c in enumerate(chunks))
    crit_keys   = ", ".join(
        f'"chunk_{i+1}": {{"verdict": "relevant|not_relevant|uncertain", "reason": "..."}}'
        for i in range(len(chunks))
    )
    return textwrap.dedent(f"""
        You are evaluating whether retrieved text chunks are relevant to a question.

        QUESTION:
        {question}

        {chunk_block}

        For each chunk, decide: relevant | not_relevant | uncertain

        Respond with ONLY this JSON object:
        {{
          {crit_keys}
        }}
    """).strip()


def _chunk_relevance_individual_prompt(question: str, chunk: str) -> str:
    return textwrap.dedent(f"""
        You are evaluating whether a retrieved text chunk is relevant to a question.

        QUESTION:
        {question}

        RETRIEVED CHUNK:
        {chunk}

        Respond with ONLY this JSON object:
        {{
          "verdict": "relevant" | "not_relevant" | "uncertain",
          "reason": "<one sentence>"
        }}
    """).strip()


def _faithfulness_prompt(answer: str, chunks: list[str]) -> str:
    chunk_block = "\n\n".join(f"[CHUNK {i+1}]: {c}" for i, c in enumerate(chunks))
    return textwrap.dedent(f"""
        You are evaluating whether an answer is faithful to retrieved text chunks.
        Faithful means every factual claim in the answer can be traced to the chunks.

        RETRIEVED CHUNKS:
        {chunk_block}

        ANSWER:
        {answer}

        Respond with ONLY this JSON object:
        {{
          "verdict": "faithful" | "partially_faithful" | "unfaithful",
          "unsupported_claims": ["<claim not in chunks>"],
          "explanation": "<2-3 sentences>"
        }}
    """).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Judge runner
# ─────────────────────────────────────────────────────────────────────────────

def run_judges(
    item:             dict,
    ts_answer:        str,
    chunks_info:      list[dict],
    judge:            JudgeClient,
    chunk_judge_mode: str = "group",
    rubric_judge_mode: str = "all",
    skip_faithfulness: bool = False
) -> dict:
    """
    Run all judge evaluations for one benchmark item.
    Returns the full judgements dict.
    """
    question      = item["question"]
    mock_answer   = item["answer"]
    must_rubric   = item["must_rubric"]
    opt_rubric    = item["optional_rubric"]
    chunk_texts   = [c.get("content", "") for c in chunks_info]

    j: dict = {}

    # ── Must rubric ───────────────────────────────────────────────────────────
    print(f"      [JUDGE] Must rubric ({rubric_judge_mode} mode) ...", flush=True)
    if rubric_judge_mode == "individual":
        must_results = []
        for crit in must_rubric:
            parsed = judge.call(_rubric_individual_prompt(question, ts_answer, crit, False))
            must_results.append({
                "criterion": crit,
                "verdict":   safe_verdict(parsed, "verdict", {"met","not_met","partial"}, "not_met"),
                "reason":    (parsed or {}).get("reason", ""),
            })
        j["must_rubric_satisfaction"] = must_results
    else:  # all
        if must_rubric:
            parsed = judge.call(_rubric_all_prompt(question, ts_answer, must_rubric, "required"))
            j["must_rubric_satisfaction"] = [
                {
                    "criterion": crit,
                    "verdict": safe_verdict(
                        (parsed or {}).get(f"criterion_{i+1}", {}),
                        "verdict", {"met","not_met","partial"}, "not_met"
                    ),
                    "reason": ((parsed or {}).get(f"criterion_{i+1}") or {}).get("reason", ""),
                }
                for i, crit in enumerate(must_rubric)
            ]
        else:
            j["must_rubric_satisfaction"] = []

    # ── Optional rubric ───────────────────────────────────────────────────────
    print(f"      [JUDGE] Optional rubric ...", flush=True)
    if opt_rubric:
        if rubric_judge_mode == "individual":
            opt_results = []
            for crit in opt_rubric:
                parsed = judge.call(_rubric_individual_prompt(question, ts_answer, crit, True))
                opt_results.append({
                    "criterion": crit,
                    "verdict":   safe_verdict(parsed, "verdict", {"met","not_met","partial"}, "not_met"),
                    "reason":    (parsed or {}).get("reason", ""),
                })
            j["optional_rubric_satisfaction"] = opt_results
        else:
            parsed = judge.call(_rubric_all_prompt(question, ts_answer, opt_rubric, "optional"))
            j["optional_rubric_satisfaction"] = [
                {
                    "criterion": crit,
                    "verdict": safe_verdict(
                        (parsed or {}).get(f"criterion_{i+1}", {}),
                        "verdict", {"met","not_met","partial"}, "not_met"
                    ),
                    "reason": ((parsed or {}).get(f"criterion_{i+1}") or {}).get("reason", ""),
                }
                for i, crit in enumerate(opt_rubric)
            ]
    else:
        j["optional_rubric_satisfaction"] = []

    # ── Answer correctness ────────────────────────────────────────────────────
    print(f"      [JUDGE] Answer correctness ...", flush=True)

    def _extract_correctness(parsed: Optional[dict]) -> dict:
        if not parsed:
            return {"score": 0, "explanation": "", "strengths": [], "weaknesses": []}
        raw_score = parsed.get("score", 0)
        try:
            score = max(-1, min(1, int(raw_score)))
        except (ValueError, TypeError):
            score = 0
        return {
            "score":       score,
            "explanation": parsed.get("explanation", ""),
            "strengths":   parsed.get("strengths", []),
            "weaknesses":  parsed.get("weaknesses", []),
        }

    p_no  = judge.call(_correctness_prompt(question, ts_answer, must_rubric, None))
    p_ref = judge.call(_correctness_prompt(question, ts_answer, must_rubric, mock_answer))
    j["answer_correctness"] = {
        "without_reference": _extract_correctness(p_no),
        "with_reference":    _extract_correctness(p_ref),
    }

    # ── Chunk relevance ───────────────────────────────────────────────────────
    print(f"      [JUDGE] Chunk relevance ({chunk_judge_mode} mode) ...", flush=True)
    individual_results: list[dict] = []
    group_results:      list[dict] = []

    if chunk_judge_mode == "individual":
        for ch in chunks_info:
            parsed = judge.call(
                _chunk_relevance_individual_prompt(question, ch.get("content", ""))
            )
            individual_results.append({
                "chunk_id": ch.get("chunk_id"),
                "rank":     ch.get("rank"),
                "verdict":  safe_verdict(parsed, "verdict",
                                         {"relevant","not_relevant","uncertain"}, "uncertain"),
                "reason":   (parsed or {}).get("reason", ""),
            })
    else:  # group
        for start in range(0, len(chunks_info), 3):
            grp    = chunks_info[start: start + 3]
            texts  = [g.get("content", "") for g in grp]
            parsed = judge.call(_chunk_relevance_group_prompt(question, texts))
            for gi, ch in enumerate(grp):
                cp = (parsed or {}).get(f"chunk_{gi+1}", {})
                group_results.append({
                    "chunk_id": ch.get("chunk_id"),
                    "rank":     ch.get("rank"),
                    "verdict":  safe_verdict(cp if isinstance(cp, dict) else {},
                                             "verdict",
                                             {"relevant","not_relevant","uncertain"}, "uncertain"),
                    "reason":   cp.get("reason", "") if isinstance(cp, dict) else "",
                })

    j["chunk_relevance"] = {"individual": individual_results, "group": group_results}

    # ── Faithfulness ──────────────────────────────────────────────────────────
    if skip_faithfulness:
        j["faithfulness"] = {
            "verdict": "skipped",
            "unsupported_claims": [],
            "explanation": "Faithfulness evaluation was skipped.",
        }
    elif chunk_texts:
        print(f"      [JUDGE] Faithfulness ...", flush=True)
        parsed = judge.call(_faithfulness_prompt(ts_answer, chunk_texts))
        j["faithfulness"] = {
            "verdict":            safe_verdict(parsed, "verdict",
                                  {"faithful","partially_faithful","unfaithful","uncertain"},
                                  "uncertain"),
            "unsupported_claims": (parsed or {}).get("unsupported_claims", []),
            "explanation":        (parsed or {}).get("explanation", ""),
        }
    else:
        j["faithfulness"] = {
            "verdict": "uncertain",
            "unsupported_claims": [],
            "explanation": "No chunks retrieved.",
        }

    return j


# ─────────────────────────────────────────────────────────────────────────────
# Summary computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_summary(judgements: dict, n_must: int, n_opt: int) -> dict:
    """Compute scalar summary metrics from a judgements dict."""
    s: dict = {}

    # Must rubric — counts toward overall
    must_res = judgements.get("must_rubric_satisfaction", [])
    if must_res:
        n_met = sum(1 for r in must_res if r.get("verdict") == "met")
        s["must_rubric_met_rate"]   = n_met / len(must_res)
        s["must_rubric_met_count"]  = n_met
        s["must_rubric_total"]      = len(must_res)
    else:
        s["must_rubric_met_rate"]   = None
        s["must_rubric_met_count"]  = 0
        s["must_rubric_total"]      = n_must

    # Optional rubric — informational only
    opt_res = judgements.get("optional_rubric_satisfaction", [])
    if opt_res:
        n_opt_met = sum(1 for r in opt_res if r.get("verdict") == "met")
        s["opt_rubric_met_rate"]    = n_opt_met / len(opt_res)
        s["opt_rubric_met_count"]   = n_opt_met
        s["opt_rubric_total"]       = len(opt_res)
    else:
        s["opt_rubric_met_rate"]    = None
        s["opt_rubric_met_count"]   = 0
        s["opt_rubric_total"]       = n_opt

    # Answer correctness
    corr = judgements.get("answer_correctness", {})
    s["correctness_with_ref"]    = corr.get("with_reference", {}).get("score")
    s["correctness_no_ref"]      = corr.get("without_reference", {}).get("score")

    # Faithfulness
    fv = judgements.get("faithfulness", {}).get("verdict", "uncertain")
    s["faithfulness_verdict"] = fv
    s["faithfulness_score"]   = {"faithful": 1.0, "partially_faithful": 0.5,
                                  "unfaithful": 0.0, "uncertain": None}.get(fv)

    # Chunk relevance
    rel = (
        judgements.get("chunk_relevance", {}).get("group")
        or judgements.get("chunk_relevance", {}).get("individual")
        or []
    )
    if rel:
        valid_rel = [r for r in rel if r.get("verdict") in ("relevant", "not_relevant")]
        if valid_rel:
            s["chunk_relevance_rate"] = sum(
                1 for r in valid_rel if r["verdict"] == "relevant"
            ) / len(valid_rel)
        else:
            s["chunk_relevance_rate"] = None
    else:
        s["chunk_relevance_rate"] = None

    return s


# ─────────────────────────────────────────────────────────────────────────────
# Report generation
# ─────────────────────────────────────────────────────────────────────────────

def _pct(v) -> str:
    if v is None: return "N/A"
    return f"{v*100:.1f}%"


def _f(v, d=2) -> str:
    if v is None: return "N/A"
    return f"{v:.{d}f}"


def _score_label(v) -> str:
    return {1: "✅ Fully correct", 0: "⚠️ Partially correct", -1: "❌ Incorrect"}.get(v, "N/A")


def _faith_label(v) -> str:
    return {
        "faithful": "✅ Faithful",
        "partially_faithful": "⚠️ Partially faithful",
        "unfaithful": "❌ Unfaithful",
        "uncertain": "❓ Uncertain",
    }.get(v or "", "❓ Unknown")


def _table(headers: list, rows: list) -> str:
    sep  = "| " + " | ".join("---" for _ in headers) + " |"
    head = "| " + " | ".join(headers) + " |"
    body = "\n".join("| " + " | ".join(str(c) for c in row) + " |" for row in rows)
    return "\n".join([head, sep, body])


def generate_report(
    label:       str,
    benchmark_path: str,
    judge_desc:  str,
    config_state: dict,
    results:     list[dict],
    output_path: pathlib.Path,
) -> None:
    lines = []
    W = lines.append

    W(f"# TokenSmith External Benchmark Report")
    W(f"")
    W(f"**Run label:** `{label}`  ")
    W(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}  ")
    W(f"**Benchmark file:** `{benchmark_path}`  ")
    W(f"**Questions evaluated:** {len(results)}  ")
    W(f"**Judge:** `{judge_desc}`  ")
    W(f"")
    W(f"---")
    W(f"")

    # ── Config ────────────────────────────────────────────────────────────────
    W(f"## ⚙️ Configuration")
    W(f"")
    W(f"```")
    for k, v in sorted(config_state.items()):
        W(f"  {k}: {v}")
    W(f"```")
    W(f"")

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    def _col(key):
        return [r["summary"].get(key) for r in results if r.get("summary")]

    def _mean(vals):
        valid = [v for v in vals if v is not None]
        return sum(valid) / len(valid) if valid else None

    n = len(results)
    must_rate   = _mean(_col("must_rubric_met_rate"))
    opt_rate    = _mean(_col("opt_rubric_met_rate"))
    corr_ref    = _mean(_col("correctness_with_ref"))
    corr_noref  = _mean(_col("correctness_no_ref"))
    faith_score = _mean(_col("faithfulness_score"))
    chunk_rel   = _mean(_col("chunk_relevance_rate"))

    W(f"---")
    W(f"")
    W(f"## 📊 Overall Results")
    W(f"")
    W(_table(
        ["Metric", "Score", "Notes"],
        [
            ["Must Rubric Met Rate", _pct(must_rate),
             "Fraction of required rubric criteria fully met"],
            ["Optional Rubric Met Rate", _pct(opt_rate),
             "Informational only — not counted in overall score"],
            ["Answer Correctness (with ref)", _f(corr_ref),
             "Mean -1/0/1 score judged with mock answer as reference"],
            ["Answer Correctness (no ref)", _f(corr_noref),
             "Mean -1/0/1 score judged without reference"],
            ["Answer Faithfulness", _f(faith_score),
             "1=faithful, 0.5=partial, 0=unfaithful"],
            ["Chunk Relevance Rate", _pct(chunk_rel),
             "Fraction of retrieved chunks judged relevant to the question"],
        ]
    ))
    W(f"")

    # Correctness distribution
    corr_dist = {1: 0, 0: 0, -1: 0}
    for r in results:
        sc = (r.get("summary") or {}).get("correctness_with_ref")
        if sc in corr_dist:
            corr_dist[sc] += 1

    W(f"### Answer Correctness Distribution (with reference)")
    W(f"")
    W(_table(
        ["Score", "Count", "Percentage"],
        [
            ["✅ 1 (Fully correct)",    str(corr_dist[1]),  _pct(corr_dist[1]/n if n else 0)],
            ["⚠️ 0 (Partially correct)", str(corr_dist[0]),  _pct(corr_dist[0]/n if n else 0)],
            ["❌ -1 (Incorrect)",       str(corr_dist[-1]), _pct(corr_dist[-1]/n if n else 0)],
        ]
    ))
    W(f"")

    # Faithfulness distribution
    W(f"### Faithfulness Distribution")
    W(f"")
    faith_dist: dict[str, int] = {}
    for r in results:
        fv = (r.get("summary") or {}).get("faithfulness_verdict", "uncertain")
        faith_dist[fv] = faith_dist.get(fv, 0) + 1

    faith_rows = []
    for v in ("faithful", "partially_faithful", "unfaithful", "uncertain"):
        cnt = faith_dist.get(v, 0)
        faith_rows.append([_faith_label(v), str(cnt), _pct(cnt/n if n else 0)])
    W(_table(["Verdict", "Count", "Percentage"], faith_rows))
    W(f"")

    # ── Per-question breakdown ─────────────────────────────────────────────────
    W(f"---")
    W(f"")
    W(f"## 📝 Per-Question Breakdown")
    W(f"")

    for r in results:
        item  = r["benchmark_item"]
        j     = r.get("judgements", {})
        s     = r.get("summary", {})
        q_id  = item.get("id", "?")
        ans   = re.sub(r"<<<[A-Z_]+>>>", "", r.get("ts_answer", "")).strip()

        W(f"### Question `{q_id}`")
        W(f"")
        W(f"**Q:** {item['question']}")
        W(f"")

        # Scorecards
        sc_w   = s.get("correctness_with_ref")
        sc_n   = s.get("correctness_no_ref")
        fv_str = _faith_label(s.get("faithfulness_verdict"))
        nm, nt = s.get("must_rubric_met_count", 0), s.get("must_rubric_total", 0)
        om, ot = s.get("opt_rubric_met_count", 0), s.get("opt_rubric_total", 0)

        W(f"| Must Rubric | Opt Rubric | Correct (ref) | Correct (no ref) | Faithfulness |")
        W(f"| --- | --- | --- | --- | --- |")
        W(f"| {nm}/{nt} | {om}/{ot} | {_score_label(sc_w)} | {_score_label(sc_n)} | {fv_str} |")
        W(f"")

        W(f"**TokenSmith Answer:**")
        W(f"")
        W(f"> {ans.replace(chr(10), '  ' + chr(10) + '> ')}")
        W(f"")

        # Must rubric
        must_res = j.get("must_rubric_satisfaction", [])
        if must_res:
            W(f"**Must Rubric:**")
            W(f"")
            for rr in must_res:
                vd  = rr.get("verdict", "")
                icon = {"met": "✅", "partial": "⚠️", "not_met": "❌"}.get(vd, "❓")
                W(f"- {icon} `{vd.upper()}` — {rr['criterion']}")
                if rr.get("reason"):
                    W(f"  - *{rr['reason']}*")
            W(f"")

        # Optional rubric
        opt_res = j.get("optional_rubric_satisfaction", [])
        if opt_res:
            W(f"**Optional Rubric:**")
            W(f"")
            for rr in opt_res:
                vd   = rr.get("verdict", "")
                icon = {"met": "✅", "partial": "⚠️", "not_met": "❌"}.get(vd, "❓")
                W(f"- {icon} `{vd.upper()}` — {rr['criterion']}")
                if rr.get("reason"):
                    W(f"  - *{rr['reason']}*")
            W(f"")

        # Correctness explanation
        corr = j.get("answer_correctness", {})
        exp_ref = corr.get("with_reference", {}).get("explanation", "")
        if exp_ref:
            W(f"**Judge explanation (with ref):** {exp_ref}")
            W(f"")

        # Faithfulness
        faith = j.get("faithfulness", {})
        if faith.get("explanation"):
            W(f"**Faithfulness:** {faith['explanation']}")
            W(f"")

        unsup = faith.get("unsupported_claims", [])
        if unsup:
            W(f"**Unsupported claims:**")
            for claim in unsup:
                W(f"- {claim}")
            W(f"")

        # Retrieved chunks summary
        chunks = r.get("retrieved_chunks", [])
        rel_items = (
            j.get("chunk_relevance", {}).get("group")
            or j.get("chunk_relevance", {}).get("individual")
            or []
        )
        rel_by_rank = {rr.get("rank"): rr for rr in rel_items}
        if chunks:
            W(f"**Retrieved chunks ({len(chunks)}):**")
            W(f"")
            for ch in chunks:
                rk   = ch.get("rank", "?")
                rr   = rel_by_rank.get(rk, {})
                vd   = rr.get("verdict", "")
                icon = {"relevant": "✅", "not_relevant": "❌", "uncertain": "❓"}.get(vd, "  ")
                preview = ch.get("content", "")[:100].replace("\n", " ")
                W(f"- {icon} **Rank {rk}** — `{preview}…`")
            W(f"")

        W(f"---")
        W(f"")

    W(f"*Report generated by TokenSmith External Benchmark Evaluator — "
      f"{time.strftime('%Y-%m-%d %H:%M:%S')}*")
    W(f"")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  [REPORT] Saved: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(args: argparse.Namespace) -> None:
    # ── Config ────────────────────────────────────────────────────────────────
    cfg_path = pathlib.Path(args.config)
    if not cfg_path.exists():
        print(f"ERROR: Config not found: {cfg_path}", file=sys.stderr)
        sys.exit(1)
    cfg = RAGConfig.from_yaml(cfg_path)

    # ── Output directory ──────────────────────────────────────────────────────
    label   = args.label or f"ext_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = pathlib.Path(args.output_dir) / label
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Benchmark ─────────────────────────────────────────────────────────────
    benchmark = load_benchmark(pathlib.Path(args.benchmark))

    print(f"\n{'='*60}")
    print(f"EXTERNAL BENCHMARK RUN")
    print(f"{'='*60}")
    print(f"  Label       : {label}")
    print(f"  Benchmark   : {args.benchmark}")
    print(f"  Questions   : {len(benchmark)}")
    print(f"  Config      : {args.config}")
    print(f"  Output      : {out_dir}")
    print(f"  Reusing run : {args.reuse_run or 'no — running TokenSmith fresh'}")
    print(f"  Judge       : {'disabled' if args.no_judge else args.judge_backend}")
    if not args.no_judge and args.judge_backend == "openrouter":
        print(f"  Judge model : {args.judge_model}")

    if args.dry_run:
        print(f"\n[DRY RUN] No execution performed.")
        return

    # ── Judge client ──────────────────────────────────────────────────────────
    judge = None
    if not args.no_judge:
        if args.judge_backend == "openrouter":
            or_key = args.openrouter_key or os.environ.get("OPENROUTER_API_KEY", "")
            if not or_key:
                print(
                    "ERROR: OpenRouter API key required for openrouter backend.\n"
                    "  Set OPENROUTER_API_KEY in .env or pass --openrouter-key.",
                    file=sys.stderr,
                )
                sys.exit(1)
            judge = JudgeClient(
                backend="openrouter",
                openrouter_key=or_key,
                openrouter_model=args.judge_model,
            )
        else:
            judge = JudgeClient(
                backend="local",
                local_model_path=cfg.gen_model,
            )
        print(f"  Judge desc  : {judge.describe()}")

    # ── TokenSmith artifacts ──────────────────────────────────────────────────
    print(f"\nLoading TokenSmith artifacts ...")
    artifacts = load_ts_artifacts(cfg)
    logger    = get_logger()
    print(f"  OK — {len(artifacts['chunks'])} chunks loaded")

    # ── Load existing results for crash recovery ──────────────────────────────
    full_path = out_dir / "full_results.json"
    existing: dict[str, dict] = {}
    if not args.no_resume and full_path.exists():
        with open(full_path, encoding="utf-8") as f:
            for rec in json.load(f):
                existing[rec["id"]] = rec
        if existing:
            print(f"  Resuming — {len(existing)} results already exist")

    # ── Load previous run if reusing ──────────────────────────────────────────
    previous_results = load_previous_ts_results(args.reuse_run) if args.reuse_run else {}

    # ── Main loop ─────────────────────────────────────────────────────────────
    results:  list[dict] = list(existing.values())
    done_ids: set[str]   = set(existing.keys())

    for i, item in enumerate(benchmark):
        qid = item["id"]
        if qid in done_ids:
            print(f"  [{i+1:>3}/{len(benchmark)}] {qid} — already done, skipping")
            continue

        question = item["question"]
        print(
            f"\n  [{i+1:>3}/{len(benchmark)}] {qid} — "
            f"{question[:55]}...",
            flush=True,
        )

        # ── TokenSmith or reuse ───────────────────────────────────────────────
        if args.reuse_run:
            prev = previous_results.get(qid)
            if prev:
                ts_answer   = prev.get("ts_answer", "")
                chunks_info = prev.get("retrieved_chunks", [])
                hyde_query  = prev.get("hyde_query")
                print(f"    Reusing previous run — {len(chunks_info)} chunks")
            else:
                print(f"    [WARN] Q {qid} not found in previous run — running TokenSmith")
                ts_answer, chunks_info, hyde_query = run_question_through_ts(
                    question, cfg, artifacts, logger
                )
                print(f"    TokenSmith: {len(chunks_info)} chunks retrieved")
        else:
            ts_answer, chunks_info, hyde_query = run_question_through_ts(
                question, cfg, artifacts, logger
            )
            print(f"    TokenSmith: {len(chunks_info)} chunks retrieved")

        # ── Run judges ────────────────────────────────────────────────────────
        judgements: dict = {}
        if judge is not None:
            judgements = run_judges(
                item=item,
                ts_answer=ts_answer,
                chunks_info=chunks_info,
                judge=judge,
                chunk_judge_mode=args.chunk_judge_mode,
                rubric_judge_mode=args.rubric_judge_mode,
                skip_faithfulness=args.no_faithfulness,
            )

        summary = compute_summary(
            judgements,
            n_must=len(item["must_rubric"]),
            n_opt=len(item["optional_rubric"]),
        )

        rec = {
            "id":               qid,
            "benchmark_item":   item,
            "ts_answer":        ts_answer,
            "retrieved_chunks": chunks_info,
            "hyde_query":       hyde_query,
            "judgements":       judgements,
            "summary":          summary,
            "judge_backend":    judge.describe() if judge else "none",
            "config_state":     cfg.get_config_state(),
            "timestamp":        time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        results.append(rec)
        done_ids.add(qid)

        # Save after each question — crash safe
        with open(full_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        time.sleep(0.3)

    # ── Final saves ───────────────────────────────────────────────────────────
    print(f"\n  Saving results ...")

    raw_records = [
        {
            "id":               r["id"],
            "question":         r["benchmark_item"]["question"],
            "ts_answer":        r["ts_answer"],
            "retrieved_chunks": r["retrieved_chunks"],
            "hyde_query":       r["hyde_query"],
        }
        for r in results
    ]
    with open(out_dir / "raw_results.json", "w", encoding="utf-8") as f:
        json.dump(raw_records, f, indent=2, ensure_ascii=False)

    judge_records = [
        {"id": r["id"], "judgements": r["judgements"], "summary": r["summary"]}
        for r in results
    ]
    with open(out_dir / "judge_results.json", "w", encoding="utf-8") as f:
        json.dump(judge_records, f, indent=2, ensure_ascii=False)

    print(f"  [SAVED] full_results.json  ({len(results)} records)")
    print(f"  [SAVED] raw_results.json")
    print(f"  [SAVED] judge_results.json")

    judge_desc = judge.describe() if judge else "none"
    generate_report(
        label=label,
        benchmark_path=args.benchmark,
        judge_desc=judge_desc,
        config_state=cfg.get_config_state(),
        results=results,
        output_path=out_dir / "report.md",
    )

    summaries = [r["summary"] for r in results if r.get("summary")]
    def _avg(key):
        vals = [s[key] for s in summaries if s.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY — {label}")
    print(f"{'='*60}")
    print(f"  Must rubric met rate     : {(_avg('must_rubric_met_rate') or 0)*100:.1f}%")
    print(f"  Optional rubric met rate : {(_avg('opt_rubric_met_rate') or 0)*100:.1f}%")
    print(f"  Correctness (with ref)   : {_avg('correctness_with_ref') or 0:.2f} / 1.0")
    print(f"  Correctness (no ref)     : {_avg('correctness_no_ref') or 0:.2f} / 1.0")
    print(f"  Faithfulness score       : {_avg('faithfulness_score') or 0:.2f} / 1.0")
    print(f"  Chunk relevance rate     : {(_avg('chunk_relevance_rate') or 0)*100:.1f}%")
    print(f"  Output                   : {out_dir}")
    print(f"{'='*60}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 src/benchmark_eval/run_external_benchmark.py",
        description="Run an external benchmark JSON file through TokenSmith + LLM judge.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--benchmark", required=True,
        help="Path to the benchmark JSON file",
    )
    parser.add_argument(
        "--config", default=DEFAULT_CONFIG,
        help=f"Path to TokenSmith config.yaml (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--output-dir", dest="output_dir", default=DEFAULT_OUT_ROOT,
        help=f"Root directory for output files (default: {DEFAULT_OUT_ROOT})",
    )
    parser.add_argument(
        "--label", default=None,
        help="Run label (default: auto-generated from timestamp)",
    )
    parser.add_argument(
        "--judge-backend", dest="judge_backend",
        choices=["local", "openrouter"], default="local",
        help="Judge backend: local Qwen GGUF (default) or OpenRouter API",
    )
    parser.add_argument(
        "--judge-model", dest="judge_model",
        default="qwen/qwen-2.5-72b-instruct",
        help="OpenRouter model ID for judge (only used with --judge-backend openrouter)",
    )
    parser.add_argument(
        "--openrouter-key", dest="openrouter_key", default="",
        help="OpenRouter API key (overrides .env / environment variable)",
    )
    parser.add_argument(
        "--chunk-judge-mode", dest="chunk_judge_mode",
        choices=["individual", "group"], default="group",
        help="How to judge chunk relevance (default: group — 3 chunks per call)",
    )
    parser.add_argument(
        "--rubric-judge-mode", dest="rubric_judge_mode",
        choices=["individual", "all"], default="all",
        help="How to judge rubric satisfaction (default: all — one call for all criteria)",
    )
    parser.add_argument(
        "--no-judge", dest="no_judge", action="store_true",
        help="Skip all LLM judge evaluations — retrieve only",
    )
    parser.add_argument(
        "--no-resume", dest="no_resume", action="store_true",
        help="Rerun everything from scratch, ignoring existing results",
    )
    parser.add_argument(
        "--dry-run", dest="dry_run", action="store_true",
        help="Print execution plan without running anything",
    )
    parser.add_argument(
        "--no-faithfulness", dest="no_faithfulness", action="store_true",
        help="Skip faithfulness judge evaluation",
    )
    parser.add_argument(
        "--reuse-run", dest="reuse_run", default=None,
        metavar="RUN_LABEL",
        help=(
            "Reuse TokenSmith answers and retrieved chunks from a previous run. "
            "Pass the run label (directory name under benchmark_results/). "
            "Only the judge evaluations will run — TokenSmith is skipped entirely."
    ),
)
    return parser


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()