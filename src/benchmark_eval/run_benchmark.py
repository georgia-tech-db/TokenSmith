#!/usr/bin/env python3
"""
src/benchmark_eval/run_benchmark.py

CLI entry point for TokenSmith benchmark evaluation.

Sub-commands
------------
run
    Run the full benchmark on one or more verified QAC files.
    Produces: raw_results.jsonl, metrics_detail.csv,
              metrics_summary.json, report.md

ab-test
    Grid-search over one or more config parameter values.
    Produces per-combination results plus a comparative_report.md.

Usage examples
--------------
    # Single run, all approved QACs in one file
    python src/benchmark_eval/run_benchmark.py run \\
        --qac_file synthetic_qac_data/manually_verified/Verified--chapter_08_qac.jsonl \\
        --config   config/config.yaml \\
        --label    baseline

    # Single run, skip judge (retrieval metrics only)
    python src/benchmark_eval/run_benchmark.py run \\
        --qac_file ... --no_judge

    # AB test: vary top_k and rerank_mode
    python src/benchmark_eval/run_benchmark.py ab-test \\
        --qac_file ... \\
        --params   top_k=5,10,20 rerank_mode=cross_encoder, \\
        --label    topk_vs_rerank

    # Dry run (print plan without running anything)
    python src/benchmark_eval/run_benchmark.py run \\
        --qac_file ... --dry_run
"""

from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import sys
import time
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

# ── Project root on path ──────────────────────────────────────────────────────
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import yaml

from src.config import RAGConfig
from src.benchmark_eval.runner import (
    INVALID_AB_PARAMS,
    load_tokensmith_artifacts,
    load_verified_qacs,
    run_benchmark,
)
from src.benchmark_eval.judge import run_all_judges
from src.benchmark_eval.metrics import (
    build_full_metrics,
    extract_per_qac_metrics,
    find_examples,
    save_metrics_csv,
    save_metrics_json,
)
from src.benchmark_eval.report import generate_ab_report, generate_report


# ─────────────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG   = "config/config.yaml"
DEFAULT_OUT_ROOT = "benchmark_results"
INDEX_PREFIX     = "textbook_index"
AB_COMBO_WARNING = 8     # warn if more than this many combinations


# ─────────────────────────────────────────────────────────────────────────────
# Config helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_cfg(config_path: str) -> RAGConfig:
    path = pathlib.Path(config_path)
    if not path.exists():
        print(f"ERROR: Config not found: {path}", file=sys.stderr)
        sys.exit(1)
    return RAGConfig.from_yaml(path)


def apply_param_override(cfg: RAGConfig, param: str, value: Any) -> None:
    """
    Apply one param override to a RAGConfig instance in-place.
    Validates that the param is not in INVALID_AB_PARAMS and that it exists.
    """
    if param in INVALID_AB_PARAMS:
        print(
            f"\nERROR: Parameter '{param}' requires re-indexing or re-embedding "
            f"and cannot be varied at query time.\n"
            f"  AB testing only supports query-time parameters such as:\n"
            f"  top_k, num_candidates, rerank_mode, rerank_top_k,\n"
            f"  ranker_weights, ensemble_method, rrf_k,\n"
            f"  gen_model, max_gen_tokens, system_prompt_mode,\n"
            f"  use_hyde, hyde_max_tokens, use_double_prompt\n",
            file=sys.stderr,
        )
        sys.exit(1)

    if not hasattr(cfg, param):
        print(
            f"\nERROR: '{param}' is not a valid RAGConfig parameter.\n"
            f"  Valid query-time parameters: top_k, num_candidates, rerank_mode,\n"
            f"  rerank_top_k, ranker_weights, ensemble_method, rrf_k, gen_model,\n"
            f"  max_gen_tokens, system_prompt_mode, use_hyde, hyde_max_tokens,\n"
            f"  use_double_prompt\n",
            file=sys.stderr,
        )
        sys.exit(1)

    # Type coercion: match the existing attribute type
    existing = getattr(cfg, param)
    try:
        if isinstance(existing, bool):
            coerced = str(value).lower() in ("true", "1", "yes")
        elif isinstance(existing, int):
            coerced = int(value)
        elif isinstance(existing, float):
            coerced = float(value)
        elif isinstance(existing, dict):
            if isinstance(value, dict):
                coerced = value
            else:
                coerced = json.loads(value)
        else:
            coerced = value
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        print(
            f"\nERROR: Could not coerce '{value}' to the type of '{param}' "
            f"(expected {type(existing).__name__}): {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    setattr(cfg, param, coerced)
    # Re-run post-init validation to catch constraint violations
    try:
        cfg.__post_init__()
    except (AssertionError, ValueError) as exc:
        print(f"\nERROR: Invalid value '{coerced}' for '{param}': {exc}", file=sys.stderr)
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Parameter grid parsing   --params top_k=5,10,20 rerank_mode=cross_encoder,
# ─────────────────────────────────────────────────────────────────────────────

def parse_param_grid(param_specs: List[str]) -> Dict[str, List]:
    """
    Parse --params arguments like "top_k=5,10,20" or "rerank_mode=cross_encoder,"
    into a dict of {param_name: [value1, value2, ...]}.

    Trailing commas are ignored.
    """
    grid: Dict[str, List] = {}
    for spec in param_specs:
        if "=" not in spec:
            print(
                f"ERROR: Invalid --params spec '{spec}'. "
                f"Expected format: param_name=val1,val2,val3",
                file=sys.stderr,
            )
            sys.exit(1)
        param, raw_values = spec.split("=", 1)
        param  = param.strip()
        values = [v.strip() for v in raw_values.split(",") if v.strip()]
        if not values:
            print(f"ERROR: No values for param '{param}'.", file=sys.stderr)
            sys.exit(1)
        grid[param] = values
    return grid


def build_combinations(grid: Dict[str, List]) -> List[Dict[str, Any]]:
    """
    Build all grid-search combinations from the param grid.
    Returns a list of {param: value} dicts.
    """
    if not grid:
        return [{}]
    keys   = list(grid.keys())
    values = [grid[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def combo_label(params: Dict[str, Any]) -> str:
    if not params:
        return "default"
    return "__".join(f"{k}={v}" for k, v in sorted(params.items()))


# ─────────────────────────────────────────────────────────────────────────────
# Core evaluation loop (shared by run and ab-test)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_one_config(
    qacs:        List[Dict],
    cfg:         RAGConfig,
    artifacts:   Dict,
    out_dir:     pathlib.Path,
    label:       str,
    run_judge:   bool = True,
    resume:      bool = True,
    chunk_judge_mode: str = "group",
    rubric_judge_mode: str = "all",
) -> Tuple[List[Dict], List[Dict], List[Dict], Dict]:
    """
    Run the full evaluation pipeline for one config:
      1. Run QACs through TokenSmith  → raw_results.jsonl
      2. Run all judge evaluations   → judge_results.jsonl
      3. Compute metrics             → metrics_detail.csv + metrics_summary.json
      4. Find examples               → used by report generator

    Returns (per_qac_metrics, results, judgements, full_metrics)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_jsonl    = out_dir / "raw_results.jsonl"
    judge_jsonl  = out_dir / "judge_results.jsonl"
    csv_path     = out_dir / "metrics_detail.csv"
    json_path    = out_dir / "metrics_summary.json"

    # ── Step 1: Run through TokenSmith ────────────────────────────────────────
    results = run_benchmark(
        qacs=qacs,
        cfg=cfg,
        artifacts=artifacts,
        output_jsonl=raw_jsonl,
        run_label=label,
        resume=resume,
    )

    # ── Step 2: Judge evaluations ─────────────────────────────────────────────
    # Load existing judge results for crash recovery
    existing_judges: Dict[str, Dict] = {}
    if resume and judge_jsonl.exists():
        with open(judge_jsonl, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entry = json.loads(line)
                        rid = entry.get("record_id", "")
                        if rid:
                            existing_judges[rid] = entry["judgements"]
                    except (json.JSONDecodeError, KeyError):
                        pass
        if existing_judges:
            print(f"  [RESUME] {len(existing_judges)} judge results already exist")

    judgements: List[Dict] = []
    model_path = cfg.gen_model

    with open(judge_jsonl, "a", encoding="utf-8") as jf:
        for i, result in enumerate(results):
            rid = result.get("qac", {}).get("record_id", f"q{i:04d}")

            if rid in existing_judges:
                judgements.append(existing_judges[rid])
                continue

            if not run_judge:
                judgements.append({})
                continue

            print(
                f"\n    [JUDGE {i+1}/{len(results)}] "
                f"{result.get('qac', {}).get('question', '')[:50]} ..."
            )
            j = run_all_judges(result, model_path,
                       chunk_judge_mode=chunk_judge_mode,
                       rubric_judge_mode=rubric_judge_mode)
            judgements.append(j)
            jf.write(json.dumps({
                "record_id": rid, "judgements": j
            }, ensure_ascii=False) + "\n")

    # ── Step 3: Metrics ───────────────────────────────────────────────────────
    per_qac = [
        extract_per_qac_metrics(result, judgement)
        for result, judgement in zip(results, judgements)
    ]
    full_metrics = build_full_metrics(per_qac)

    save_metrics_csv(per_qac, csv_path)
    save_metrics_json(full_metrics, json_path)

    # ── Step 4: Examples ──────────────────────────────────────────────────────
    examples = find_examples(per_qac, results, judgements)

    return per_qac, results, judgements, full_metrics, examples


# ─────────────────────────────────────────────────────────────────────────────
# Sub-command: run
# ─────────────────────────────────────────────────────────────────────────────

def cmd_run(args: argparse.Namespace) -> None:
    cfg = load_cfg(args.config)

    # Collect QAC files
    qac_files = _resolve_qac_files(args.qac_file)
    qacs: List[Dict] = []
    for qf in qac_files:
        loaded = load_verified_qacs(qf)
        print(f"  Loaded {len(loaded)} approved QACs from {qf.name}")
        qacs.extend(loaded)

    if not qacs:
        print("ERROR: No approved QAC records found.", file=sys.stderr)
        sys.exit(1)

    label   = args.label or f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = pathlib.Path(args.output_dir) / label

    print(f"\n{'='*60}")
    print(f"BENCHMARK RUN")
    print(f"{'='*60}")
    print(f"  Label       : {label}")
    print(f"  QAC count   : {len(qacs)}")
    print(f"  Config      : {args.config}")
    print(f"  Output      : {out_dir}")
    print(f"  Judge       : {'disabled (--no_judge)' if args.no_judge else 'enabled'}")

    if args.dry_run:
        print(f"\n[DRY RUN] No evaluation performed.")
        return

    artifacts = load_tokensmith_artifacts(cfg, INDEX_PREFIX)
    print(f"  Artifacts loaded successfully")

    per_qac, results, judgements, full_metrics, examples = evaluate_one_config(
        qacs=qacs,
        cfg=cfg,
        artifacts=artifacts,
        out_dir=out_dir,
        label=label,
        run_judge=not args.no_judge,
        resume=not args.no_resume,
        chunk_judge_mode=args.chunk_judge_mode,
        rubric_judge_mode=args.rubric_judge_mode,
    )

    generate_report(
        run_label=label,
        config_state=cfg.get_config_state(),
        qac_file=str(qac_files),
        full_metrics=full_metrics,
        per_qac=per_qac,
        examples=examples,
        output_path=out_dir / "report.md",
    )

    overall = full_metrics.get("overall", {})
    print(f"\n{'='*60}")
    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  Gold chunk coverage : {(overall.get('gold_chunk_coverage_rate_mean') or 0)*100:.1f}%")
    print(f"  Rubric met rate     : {(overall.get('rubric_met_rate_mean_individual') or 0)*100:.1f}%")
    print(f"  Correctness (ref)   : {overall.get('correctness_score_with_ref_mean') or 0:.2f} / 1.0")
    print(f"  Faithfulness        : {overall.get('faithfulness_score_mean') or 0:.2f} / 1.0")
    print(f"  BLEU score          : {overall.get('bleu_score_mean') or 0:.4f}")
    print(f"  Output directory    : {out_dir}")
    print(f"{'='*60}")


# ─────────────────────────────────────────────────────────────────────────────
# Sub-command: ab-test
# ─────────────────────────────────────────────────────────────────────────────

def cmd_ab_test(args: argparse.Namespace) -> None:
    base_cfg = load_cfg(args.config)

    qac_files = _resolve_qac_files(args.qac_file)
    qacs: List[Dict] = []
    for qf in qac_files:
        loaded = load_verified_qacs(qf)
        print(f"  Loaded {len(loaded)} approved QACs from {qf.name}")
        qacs.extend(loaded)

    if not qacs:
        print("ERROR: No approved QAC records found.", file=sys.stderr)
        sys.exit(1)

    # Parse and validate param grid
    grid = parse_param_grid(args.params)

    # Validate all params upfront before doing any work
    test_cfg = deepcopy(base_cfg)
    for param, values in grid.items():
        for v in values:
            apply_param_override(test_cfg, param, v)

    combos = build_combinations(grid)
    label  = args.label or f"ab_test_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = pathlib.Path(args.output_dir) / label

    print(f"\n{'='*60}")
    print(f"AB TEST")
    print(f"{'='*60}")
    print(f"  Label       : {label}")
    print(f"  QAC count   : {len(qacs)}")
    print(f"  Parameters  : {list(grid.keys())}")
    print(f"  Combinations: {len(combos)}")
    for combo in combos:
        print(f"    - {combo_label(combo)}")
    print(f"  Judge       : {'disabled' if args.no_judge else 'enabled'}")

    if len(combos) > AB_COMBO_WARNING:
        print(
            f"\n  ⚠️  WARNING: {len(combos)} combinations is a large grid search. "
            f"This will take a significant amount of time and incur substantial "
            f"compute cost (the LLM model loads once but inference runs for each "
            f"combination × each QAC × each judge call). Consider running a subset first.\n"
        )

    if args.dry_run:
        print(f"\n[DRY RUN] No evaluation performed.")
        return

    # Load artifacts once — shared across all combinations
    artifacts = load_tokensmith_artifacts(base_cfg, INDEX_PREFIX)
    print(f"  Artifacts loaded successfully\n")

    combination_results: List[Dict] = []

    for i, combo in enumerate(combos):
        clabel = combo_label(combo)
        print(f"\n{'─'*60}")
        print(f"COMBINATION {i+1}/{len(combos)}: {clabel}")
        print(f"{'─'*60}")

        # Apply overrides to a fresh copy of the config
        cfg = deepcopy(base_cfg)
        for param, value in combo.items():
            apply_param_override(cfg, param, value)

        combo_dir = out_dir / clabel

        per_qac, results, judgements, full_metrics, examples = evaluate_one_config(
            qacs=qacs,
            cfg=cfg,
            artifacts=artifacts,
            out_dir=combo_dir,
            label=clabel,
            run_judge=not args.no_judge,
            resume=not args.no_resume,
        )

        generate_report(
            run_label=clabel,
            config_state=cfg.get_config_state(),
            qac_file=str(qac_files),
            full_metrics=full_metrics,
            per_qac=per_qac,
            examples=examples,
            output_path=combo_dir / "report.md",
        )

        combination_results.append({
            "label":        clabel,
            "params":       combo,
            "full_metrics": full_metrics,
        })

    # Generate comparative report
    generate_ab_report(
        param_grid=grid,
        combination_results=combination_results,
        output_path=out_dir / "comparative_report.md",
    )

    # Save combination summary JSON
    summary_path = out_dir / "ab_test_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(combination_results, f, indent=2, ensure_ascii=False)
    print(f"\n[AB TEST] Summary saved: {summary_path}")
    print(f"[AB TEST] Comparative report: {out_dir / 'comparative_report.md'}")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_qac_files(qac_file_arg: str) -> List[pathlib.Path]:
    """
    Resolve --qac_file to a list of paths.
    Accepts:
      - a single JSONL file path
      - a directory (uses all Verified--*.jsonl in it)
      - "all" (searches synthetic_qac_data/manually_verified/)
    """
    if qac_file_arg == "all":
        base = pathlib.Path("synthetic_qac_data") / "manually_verified"
        files = sorted(base.glob("Verified--*.jsonl"))
        if not files:
            print(
                f"ERROR: No 'Verified--*.jsonl' files found in {base}",
                file=sys.stderr,
            )
            sys.exit(1)
        return files

    path = pathlib.Path(qac_file_arg)
    if path.is_dir():
        files = sorted(path.glob("Verified--*.jsonl"))
        if not files:
            print(
                f"ERROR: No 'Verified--*.jsonl' files found in {path}",
                file=sys.stderr,
            )
            sys.exit(1)
        return files

    if not path.exists():
        print(f"ERROR: QAC file not found: {path}", file=sys.stderr)
        sys.exit(1)
    return [path]


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python src/benchmark_eval/run_benchmark.py",
        description="TokenSmith benchmark evaluation pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── Shared args ───────────────────────────────────────────────────────────
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument(
        "--qac_file", required=True,
        help=(
            "Path to a Verified--*.jsonl file, a directory containing them, "
            "or 'all' to use all files in synthetic_qac_data/manually_verified/"
        ),
    )
    shared.add_argument(
        "--config", default=DEFAULT_CONFIG,
        help=f"Path to TokenSmith config.yaml (default: {DEFAULT_CONFIG})",
    )
    shared.add_argument(
        "--output_dir", default=DEFAULT_OUT_ROOT,
        help=f"Root directory for output files (default: {DEFAULT_OUT_ROOT})",
    )
    shared.add_argument(
        "--label", default=None,
        help="Human-readable label for this run (default: auto-generated from timestamp)",
    )
    shared.add_argument(
        "--no_judge", action="store_true",
        help="Skip LLM judge evaluations (deterministic metrics only, much faster)",
    )
    shared.add_argument(
        "--no_resume", action="store_true",
        help="Do not resume from existing results — rerun everything from scratch",
    )
    shared.add_argument(
        "--dry_run", action="store_true",
        help="Print the evaluation plan without running anything",
    )
    shared.add_argument(
        "--chunk_judge_mode",
        choices=["individual", "group"],
        default="group",
        help="How to evaluate chunk relevance: one chunk at a time (individual) or 3 at a time (group). Default: group",
    )
    shared.add_argument(
        "--rubric_judge_mode",
        choices=["individual", "all"],
        default="all",
        help="How to evaluate rubric satisfaction: one criterion at a time (individual) or all together (all). Default: all",
    )

    # ── run ───────────────────────────────────────────────────────────────────
    run_p = sub.add_parser(
        "run",
        parents=[shared],
        help="Run benchmark on a verified QAC file.",
    )
    run_p.set_defaults(func=cmd_run)

    # ── ab-test ───────────────────────────────────────────────────────────────
    ab_p = sub.add_parser(
        "ab-test",
        parents=[shared],
        help="Grid-search AB test across config parameter values.",
    )
    ab_p.add_argument(
        "--params", nargs="+", required=True,
        metavar="PARAM=VAL1,VAL2,...",
        help=(
            "One or more parameter specs, e.g. "
            "'top_k=5,10,20' 'rerank_mode=cross_encoder,'. "
            "All combinations are tested (full grid search)."
        ),
    )
    ab_p.set_defaults(func=cmd_ab_test)

    return parser


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()