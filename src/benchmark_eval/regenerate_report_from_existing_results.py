# regenerate_report.py  (put this in project root or src/benchmark_eval/)
"""
Regenerate a benchmark report from existing result files without re-running
the judge or TokenSmith.

Usage:
    python3 regenerate_report.py --run_dir benchmark_results/chapter8_baseline_v2
"""

import argparse
import json
import pathlib
import sys

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.benchmark_eval.metrics import (
    build_full_metrics,
    extract_per_qac_metrics,
    find_examples,
    save_metrics_csv,
    save_metrics_json,
)
from src.benchmark_eval.report import generate_report


def load_jsonl(path: pathlib.Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate benchmark report from existing result files."
    )
    parser.add_argument(
        "--run_dir", required=True,
        help="Path to the benchmark run directory, e.g. benchmark_results/chapter8_baseline_v2"
    )
    args = parser.parse_args()

    run_dir = pathlib.Path(args.run_dir)

    raw_jsonl   = run_dir / "raw_results.jsonl"
    judge_jsonl = run_dir / "judge_results.jsonl"

    if not raw_jsonl.exists():
        print(f"ERROR: {raw_jsonl} not found.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading raw results from {raw_jsonl} ...")
    results = load_jsonl(raw_jsonl)
    print(f"  {len(results)} results loaded")

    # Load judge results — keyed by record_id
    judge_index: dict[str, dict] = {}
    if judge_jsonl.exists():
        for entry in load_jsonl(judge_jsonl):
            rid = entry.get("record_id", "")
            if rid:
                judge_index[rid] = entry.get("judgements", {})
        print(f"  {len(judge_index)} judge results loaded")
    else:
        print(f"  No judge results found — metrics will be deterministic only")

    # Align judgements to results in order
    judgements = []
    for i, result in enumerate(results):
        rid = result.get("qac", {}).get("record_id", f"q{i:04d}")
        judgements.append(judge_index.get(rid, {}))

    # Recompute all metrics
    print("Recomputing metrics ...")
    per_qac      = [extract_per_qac_metrics(r, j) for r, j in zip(results, judgements)]
    full_metrics = build_full_metrics(per_qac)
    examples     = find_examples(per_qac, results, judgements)

    # Save updated CSVs and JSON too
    save_metrics_csv(per_qac,      run_dir / "metrics_detail.csv")
    save_metrics_json(full_metrics, run_dir / "metrics_summary.json")

    # Load config state from the first result if available
    config_state = results[0].get("qac", {}) if results else {}
    # Try to get it from the summary JSON that was previously written
    summary_path = run_dir / "metrics_summary.json"
    judge_model  = ""
    if summary_path.exists():
        with open(summary_path) as f:
            saved = json.load(f)
        config_state = saved.get("config_state", config_state)
        judge_model  = saved.get("judge_model", "")

    # Generate report
    report_path = run_dir / "report.md"
    print(f"Generating report -> {report_path}")
    generate_report(
        run_label=run_dir.name,
        config_state=config_state,
        qac_file=str(run_dir),
        full_metrics=full_metrics,
        per_qac=per_qac,
        examples=examples,
        output_path=report_path,
        judge_model=judge_model,
    )
    print("Done.")


if __name__ == "__main__":
    main()