#!/usr/bin/env python3
"""
eval/run_eval.py

Evaluate the TokenSmith RAG pipeline on eval/questions.jsonl, comparing the
CompositeQueryPlanner (MultiHop -> Heuristic, the "optimizer") against a
no-op baseline planner that leaves cfg unchanged and never expands the query.

The baseline and optimizer share the same artifacts (chunks, FAISS index,
BM25 index, embedding model) — only the planner wiring differs. The eval
calls `src.main.get_answer` directly with `is_test_mode=True` so no streaming
or markdown rendering happens.

Usage:
    python -m eval.run_eval                  # run both modes (default)
    python -m eval.run_eval --baseline       # run baseline only
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
from collections import defaultdict
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

# Ensure repo root is on sys.path when invoked as a script
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import RAGConfig
from src.instrumentation.logging import get_logger
from src.main import ANSWER_NOT_FOUND, get_answer
from src.planning.composite import CompositeQueryPlanner
from src.planning.cost_model import CostModelPlanner
from src.planning.heuristics import HeuristicQueryPlanner
from src.planning.multihop import MultiHopQueryPlanner
from src.planning.noop import NoOpPlanner
from src.planning.planner import QueryPlanner
from src.ranking.ranker import EnsembleRanker
from src.retriever import (
    BM25Retriever,
    FAISSRetriever,
    IndexKeywordRetriever,
    load_artifacts,
)


INDEX_PREFIX = "textbook_index"
QUESTIONS_PATH = REPO_ROOT / "eval" / "questions.jsonl"
RESULTS_PATH = REPO_ROOT / "eval" / "results.csv"
CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"


def build_args() -> SimpleNamespace:
    # get_answer reads args.system_prompt_mode and (via getattr) args.double_prompt
    return SimpleNamespace(
        system_prompt_mode="baseline",
        double_prompt=False,
        index_prefix=INDEX_PREFIX,
    )


def build_artifacts(cfg: RAGConfig, planner: QueryPlanner) -> Dict[str, Any]:
    """
    Mirror run_chat_session's artifact setup but swap in an arbitrary planner.
    """
    artifacts_dir = cfg.get_artifacts_directory()
    faiss_idx, bm25_idx, chunks, sources, meta = load_artifacts(
        artifacts_dir, INDEX_PREFIX
    )
    retrievers: List[Any] = [
        FAISSRetriever(faiss_idx, cfg.embed_model),
        BM25Retriever(bm25_idx),
    ]
    if cfg.ranker_weights.get("index_keywords", 0) > 0:
        retrievers.append(
            IndexKeywordRetriever(cfg.extracted_index_path, cfg.page_to_chunk_map_path)
        )
    ranker = EnsembleRanker(
        ensemble_method=cfg.ensemble_method,
        weights=cfg.ranker_weights,
        rrf_k=int(cfg.rrf_k),
    )
    return {
        "chunks": chunks,
        "sources": sources,
        "retrievers": retrievers,
        "ranker": ranker,
        "meta": meta,
        "planner": planner,
    }


def load_questions() -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(QUESTIONS_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def retrieval_hit(chunks_info: Optional[List[Dict[str, Any]]], expected: List[str]) -> int:
    if not chunks_info or not expected:
        return 0
    contents = [str(c.get("content", "")).lower() for c in chunks_info]
    for needle in expected:
        n = str(needle).lower()
        if any(n in content for content in contents):
            return 1
    return 0


def answer_hit(answer_text: str, gold_fragment: str) -> int:
    if not answer_text or not gold_fragment:
        return 0
    return 1 if gold_fragment.lower() in answer_text.lower() else 0


def run_one(
    question: str,
    cfg: RAGConfig,
    artifacts: Dict[str, Any],
    args: SimpleNamespace,
    logger,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Call get_answer in test mode. Returns (answer_text, chunks_info).
    Handles the ANSWER_NOT_FOUND early-return path where get_answer yields
    a bare string instead of the tuple.
    """
    result = get_answer(
        question=question,
        cfg=cfg,
        args=args,
        logger=logger,
        console=None,
        artifacts=artifacts,
        is_test_mode=True,
    )
    if isinstance(result, tuple):
        ans, chunks_info, _ = result
        return ans or "", chunks_info or []
    # ANSWER_NOT_FOUND path — no chunks retrieved
    return str(result), []


# Empirically derived from N=116 eval (eval/results_v2_n116.csv):
# categories where the optimizer beat baseline → composite; where baseline
# won → noop. Categories not in the table fall back to the default planner.
COST_MODEL_ROUTING = {
    "keyword":     "composite",
    "definition":  "composite",
    "procedural":  "composite",
    "other":       "composite",
    "comparison":  "noop",
    "explanatory": "noop",
}


def _build_cost_model(cfg: RAGConfig) -> CostModelPlanner:
    composite = CompositeQueryPlanner(
        cfg,
        [MultiHopQueryPlanner(cfg), HeuristicQueryPlanner(cfg)],
    )
    noop = NoOpPlanner(cfg)
    table = {
        "composite": composite,
        "noop": noop,
    }
    routing = {cat: table[choice] for cat, choice in COST_MODEL_ROUTING.items()}
    return CostModelPlanner(
        cfg,
        routing_table=routing,
        default_planner=composite,
        classifier=HeuristicQueryPlanner(cfg),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="TokenSmith planner evaluation")
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Include the no-op baseline planner",
    )
    parser.add_argument(
        "--optimizer",
        action="store_true",
        help="Include the CompositeQueryPlanner (multi-hop + heuristic)",
    )
    parser.add_argument(
        "--cost-model",
        dest="cost_model",
        action="store_true",
        help="Include the CostModelPlanner (per-category routing)",
    )
    cli = parser.parse_args()

    # If no flag is passed, run all three modes. Otherwise run only the
    # explicitly requested ones.
    any_flag = cli.baseline or cli.optimizer or cli.cost_model
    run_baseline = cli.baseline or not any_flag
    run_optimizer = cli.optimizer or not any_flag
    run_cost_model = cli.cost_model or not any_flag

    if not CONFIG_PATH.exists():
        print(f"ERROR: missing config at {CONFIG_PATH}", file=sys.stderr)
        sys.exit(1)

    cfg = RAGConfig.from_yaml(CONFIG_PATH)
    logger = get_logger()
    args = build_args()

    print(f"Loading artifacts from {cfg.get_artifacts_directory()} ...")

    baseline_artifacts: Optional[Dict[str, Any]] = None
    optimizer_artifacts: Optional[Dict[str, Any]] = None
    cost_model_artifacts: Optional[Dict[str, Any]] = None
    if run_baseline:
        baseline_artifacts = build_artifacts(cfg, NoOpPlanner(cfg))
    if run_optimizer:
        composite = CompositeQueryPlanner(
            cfg,
            [MultiHopQueryPlanner(cfg), HeuristicQueryPlanner(cfg)],
        )
        optimizer_artifacts = build_artifacts(cfg, composite)
    if run_cost_model:
        cost_model_artifacts = build_artifacts(cfg, _build_cost_model(cfg))

    questions = load_questions()
    print(f"Loaded {len(questions)} questions from {QUESTIONS_PATH}")

    # Use a dedicated HeuristicQueryPlanner to label the CSV regardless of
    # which modes are run. This keeps the "planner_classification" column
    # stable and lets us diagnose misclassifications even on baseline-only
    # runs.
    label_planner = HeuristicQueryPlanner(cfg)

    rows: List[Dict[str, Any]] = []
    for i, q in enumerate(questions, 1):
        query = q["query"]
        category = q.get("category", "")
        expected = q.get("expected_chunks", [])
        gold = q.get("gold_answer_fragment", "")

        classification = label_planner.classify(query)

        b_retr: Any = ""
        o_retr: Any = ""
        c_retr: Any = ""
        b_ans: Any = ""
        o_ans: Any = ""
        c_ans: Any = ""

        print(f"\n[{i}/{len(questions)}] ({category}) {query}")

        if run_baseline and baseline_artifacts is not None:
            print("  -- baseline --")
            ans_b, chunks_b = run_one(query, cfg, baseline_artifacts, args, logger)
            b_retr = retrieval_hit(chunks_b, expected)
            b_ans = answer_hit(ans_b, gold)
            print(f"    retrieval_hit={b_retr} answer_hit={b_ans}")

        if run_optimizer and optimizer_artifacts is not None:
            print("  -- optimizer --")
            ans_o, chunks_o = run_one(query, cfg, optimizer_artifacts, args, logger)
            o_retr = retrieval_hit(chunks_o, expected)
            o_ans = answer_hit(ans_o, gold)
            print(f"    retrieval_hit={o_retr} answer_hit={o_ans}")

        if run_cost_model and cost_model_artifacts is not None:
            print("  -- cost_model --")
            ans_c, chunks_c = run_one(query, cfg, cost_model_artifacts, args, logger)
            c_retr = retrieval_hit(chunks_c, expected)
            c_ans = answer_hit(ans_c, gold)
            print(f"    retrieval_hit={c_retr} answer_hit={c_ans}")

        rows.append({
            "query": query,
            "category": category,
            "planner_classification": classification,
            "baseline_retrieval_hit": b_retr,
            "optimizer_retrieval_hit": o_retr,
            "cost_model_retrieval_hit": c_retr,
            "baseline_answer_hit": b_ans,
            "optimizer_answer_hit": o_ans,
            "cost_model_answer_hit": c_ans,
        })

    write_results(rows)
    print_summary(rows, run_baseline, run_optimizer, run_cost_model)


def write_results(rows: List[Dict[str, Any]]) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "query",
        "category",
        "planner_classification",
        "baseline_retrieval_hit",
        "optimizer_retrieval_hit",
        "cost_model_retrieval_hit",
        "baseline_answer_hit",
        "optimizer_answer_hit",
        "cost_model_answer_hit",
    ]
    with open(RESULTS_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"\nWrote results to {RESULTS_PATH}")


def _rate(values: List[int]) -> float:
    return (sum(values) / len(values)) if values else 0.0


def print_summary(
    rows: List[Dict[str, Any]],
    run_baseline: bool,
    run_optimizer: bool,
    run_cost_model: bool,
) -> None:
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        buckets[r.get("category") or "unknown"].append(r)

    cols = [f"{'category':<12}", f"{'n':>4}"]
    if run_baseline:
        cols += [f"{'B retr':>8}", f"{'B ans':>8}"]
    if run_optimizer:
        cols += [f"{'O retr':>8}", f"{'O ans':>8}"]
    if run_cost_model:
        cols += [f"{'C retr':>8}", f"{'C ans':>8}"]
    header = " ".join(cols)

    print()
    print("=" * len(header))
    print("Hit rates by category")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    def fmt_row(label: str, items: List[Dict[str, Any]]) -> str:
        parts = [f"{label:<12}", f"{len(items):>4}"]
        if run_baseline:
            br = _rate([int(x.get("baseline_retrieval_hit") or 0) for x in items])
            ba = _rate([int(x.get("baseline_answer_hit") or 0) for x in items])
            parts += [f"{br:>8.2%}", f"{ba:>8.2%}"]
        if run_optimizer:
            orr = _rate([int(x.get("optimizer_retrieval_hit") or 0) for x in items])
            oa = _rate([int(x.get("optimizer_answer_hit") or 0) for x in items])
            parts += [f"{orr:>8.2%}", f"{oa:>8.2%}"]
        if run_cost_model:
            cr = _rate([int(x.get("cost_model_retrieval_hit") or 0) for x in items])
            ca = _rate([int(x.get("cost_model_answer_hit") or 0) for x in items])
            parts += [f"{cr:>8.2%}", f"{ca:>8.2%}"]
        return " ".join(parts)

    for cat in sorted(buckets):
        print(fmt_row(cat, buckets[cat]))
    print("-" * len(header))
    print(fmt_row("ALL", rows))


if __name__ == "__main__":
    main()
