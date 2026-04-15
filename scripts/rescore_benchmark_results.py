#!/usr/bin/env python3
"""
Rescore TokenSmith benchmark results with the current retrieval-aware metric suite.

This is useful for comparing a baseline branch that produced raw benchmark outputs
before the new retrieval metrics existed. The input JSONL only needs to include:
  - test_id
  - retrieved_answer
  - chunks_info

Expected answers, keywords, retrieval gold labels, and query types are loaded from
the current benchmarks.yaml so baseline and improved runs are scored identically.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any, Dict

import yaml

from tests.metrics import SimilarityScorer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rescore benchmark results with current metrics.")
    parser.add_argument("input_results", help="Input benchmark JSONL file or results directory.")
    parser.add_argument("output_results", help="Output JSONL file path.")
    parser.add_argument(
        "--benchmarks",
        default="tests/benchmarks.yaml",
        help="Path to the benchmark definitions used for rescoring.",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=["all"],
        help="Metric configuration to pass to SimilarityScorer.",
    )
    return parser.parse_args()


def _resolve_results_path(path_str: str) -> pathlib.Path:
    path = pathlib.Path(path_str)
    if path.is_dir():
        return path / "benchmark_results.json"
    return path


def load_benchmarks(path: pathlib.Path) -> Dict[str, Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return {benchmark["id"]: benchmark for benchmark in data.get("benchmarks", [])}


def iter_results(path: pathlib.Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> None:
    args = parse_args()
    benchmark_map = load_benchmarks(pathlib.Path(args.benchmarks))
    scorer = SimilarityScorer(enabled_metrics=args.metrics)

    input_path = _resolve_results_path(args.input_results)
    output_path = pathlib.Path(args.output_results)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for result in iter_results(input_path):
            benchmark_id = result["test_id"]
            benchmark = benchmark_map.get(benchmark_id)
            if benchmark is None:
                raise KeyError(f"Benchmark '{benchmark_id}' not found in {args.benchmarks}")

            rescored = scorer.calculate_scores(
                result.get("retrieved_answer", ""),
                benchmark["expected_answer"],
                benchmark.get("keywords", []),
                question=benchmark["question"],
                retrieval_gold=benchmark.get("retrieval_gold"),
                actual_retrieved_chunks=result.get("chunks_info", []),
            )

            result["expected_answer"] = benchmark["expected_answer"]
            result["keywords"] = benchmark.get("keywords", [])
            result["retrieval_gold"] = benchmark.get("retrieval_gold")
            result["expected_query_type"] = benchmark.get("query_type")
            result["history"] = benchmark.get("history", [])
            result["scores"] = rescored
            result["passed"] = rescored.get("final_score", 0.0) >= (
                result.get("threshold") or benchmark.get("similarity_threshold", 0.6) or 0.6
            )
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"Rescored results written to {output_path}")


if __name__ == "__main__":
    main()
