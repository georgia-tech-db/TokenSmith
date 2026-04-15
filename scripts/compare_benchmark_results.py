#!/usr/bin/env python3
"""
Compare two TokenSmith benchmark result files produced by tests/test_benchmarks.py.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from collections import defaultdict
from statistics import mean
from typing import Any, Dict, Iterable, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare benchmark result JSONL files.")
    parser.add_argument("baseline", help="Baseline results file or directory.")
    parser.add_argument("improved", help="Improved results file or directory.")
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=["final_score", "chunk_ndcg_10", "page_hit_10", "chunk_recall_10"],
        help="Metrics to highlight in the per-query delta table.",
    )
    return parser.parse_args()


def _resolve_results_path(path_str: str) -> pathlib.Path:
    path = pathlib.Path(path_str)
    if path.is_dir():
        return path / "benchmark_results.json"
    return path


def load_results(path_str: str) -> Dict[str, Dict[str, Any]]:
    path = _resolve_results_path(path_str)
    results: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            results[record["test_id"]] = record
    return results


def _metric_value(result: Dict[str, Any], metric_name: str) -> float:
    scores = result.get("scores", {})
    if metric_name == "final_score":
        return float(scores.get("final_score", 0.0))
    if metric_name in scores:
        return float(scores.get(metric_name, 0.0))
    metric_key = f"{metric_name}_similarity"
    if metric_key in scores:
        return float(scores.get(metric_key, 0.0))
    return float(scores.get(metric_name, 0.0))


def summarize(results: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    aggregate: Dict[str, List[float]] = defaultdict(list)
    for result in results.values():
        scores = result.get("scores", {})
        aggregate["final_score"].append(float(scores.get("final_score", 0.0)))
        aggregate["passed"].append(1.0 if result.get("passed") else 0.0)
        for metric_name in scores.get("retrieval_metrics", []):
            aggregate[metric_name].append(_metric_value(result, metric_name))
        for metric_name in scores.get("answer_metrics", []):
            aggregate[metric_name].append(_metric_value(result, metric_name))
    return {metric_name: mean(values) for metric_name, values in aggregate.items() if values}


def confusion_pairs(results: Dict[str, Dict[str, Any]]) -> List[Tuple[str, str]]:
    pairs = []
    for result in results.values():
        expected = result.get("expected_query_type")
        actual = result.get("retrieval_trace", {}).get("resolved_query_type")
        if expected and actual:
            pairs.append((expected, actual))
    return pairs


def print_summary(label: str, summary: Dict[str, float]) -> None:
    print(label)
    for metric_name in sorted(summary):
        print(f"  {metric_name:20} {summary[metric_name]:.4f}")


def print_confusion_matrix(label: str, pairs: Iterable[Tuple[str, str]]) -> None:
    pairs = list(pairs)
    if not pairs:
        return

    expected_labels = sorted({expected for expected, _ in pairs})
    actual_labels = sorted({actual for _, actual in pairs})
    counts = {(expected, actual): 0 for expected in expected_labels for actual in actual_labels}
    for expected, actual in pairs:
        counts[(expected, actual)] += 1

    print(f"\n{label} query-type confusion matrix")
    header = "expected \\ actual".ljust(24) + "".join(actual.ljust(16) for actual in actual_labels)
    print(header)
    for expected in expected_labels:
        row = expected.ljust(24)
        row += "".join(str(counts[(expected, actual)]).ljust(16) for actual in actual_labels)
        print(row)


def print_per_query_deltas(
    baseline_results: Dict[str, Dict[str, Any]],
    improved_results: Dict[str, Dict[str, Any]],
    metric_names: List[str],
) -> None:
    common_ids = sorted(set(baseline_results) & set(improved_results))
    print("\nPer-query deltas")
    for benchmark_id in common_ids:
        print(f"\n{benchmark_id}")
        for metric_name in metric_names:
            baseline_value = _metric_value(baseline_results[benchmark_id], metric_name)
            improved_value = _metric_value(improved_results[benchmark_id], metric_name)
            delta = improved_value - baseline_value
            print(
                f"  {metric_name:18} "
                f"{baseline_value:7.4f} -> {improved_value:7.4f} "
                f"({delta:+7.4f})"
            )


def main() -> None:
    args = parse_args()
    baseline_results = load_results(args.baseline)
    improved_results = load_results(args.improved)

    baseline_summary = summarize(baseline_results)
    improved_summary = summarize(improved_results)

    print_summary("Baseline summary", baseline_summary)
    print()
    print_summary("Improved summary", improved_summary)

    print("\nAggregate deltas")
    for metric_name in sorted(set(baseline_summary) | set(improved_summary)):
        baseline_value = baseline_summary.get(metric_name, 0.0)
        improved_value = improved_summary.get(metric_name, 0.0)
        print(
            f"  {metric_name:20} {baseline_value:7.4f} -> {improved_value:7.4f} "
            f"({improved_value - baseline_value:+7.4f})"
        )

    print_confusion_matrix("Baseline", confusion_pairs(baseline_results))
    print_confusion_matrix("Improved", confusion_pairs(improved_results))
    print_per_query_deltas(baseline_results, improved_results, args.metrics)


if __name__ == "__main__":
    main()
