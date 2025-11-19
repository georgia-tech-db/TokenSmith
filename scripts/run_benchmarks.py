#!/usr/bin/env python3
"""
Execute the configured TokenSmith pipeline against a benchmark YAML file
without manual chat input. Each benchmark question is routed through the same
`get_answer` call path used by the chat CLI so logging (including latency)
works exactly as in interactive sessions.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import yaml

from src.config import QueryPlanConfig
from src.instrumentation.logging import get_logger, init_logger
from src.main import get_answer
from src.ranking.ranker import EnsembleRanker
from src.retriever import BM25Retriever, FAISSRetriever, load_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TokenSmith benchmarks through the chat pipeline."
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to the TokenSmith config file (default: %(default)s)",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["tests/benchmarks.yaml"],
        help="One or more benchmark YAML files to run sequentially (default: %(default)s)",
    )
    parser.add_argument(
        "--index_prefix",
        default="textbook_index",
        help="Index prefix to load artifacts from (default: %(default)s)",
    )
    parser.add_argument(
        "--model_path",
        default=None,
        help="Optional override for generator model path",
    )
    parser.add_argument(
        "--system_prompt_mode",
        default=None,
        help="Override system prompt mode (default: use config value)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="How many times to run the full benchmark set in this process (default: %(default)s)",
    )
    return parser.parse_args()


def load_benchmarks(path: pathlib.Path) -> List[Dict[str, Any]]:
    raw = yaml.safe_load(path.read_text())
    benches = raw.get("benchmarks", [])
    if not benches:
        raise ValueError(f"No benchmarks found in {path}")
    return benches


def prepare_artifacts(cfg: QueryPlanConfig, index_prefix: str):
    artifacts_dir = cfg.make_artifacts_directory()
    faiss_index, bm25_index, chunks, sources = load_artifacts(
        artifacts_dir=artifacts_dir,
        index_prefix=index_prefix,
    )
    retrievers = [
        FAISSRetriever(faiss_index, cfg.embed_model),
        BM25Retriever(bm25_index),
    ]
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
    }


def main() -> None:
    args = parse_args()
    config_path = pathlib.Path(args.config)
    if not config_path.exists():
        sys.exit(f"Config file not found: {config_path}")

    cfg = QueryPlanConfig.from_yaml(config_path)
    init_logger(cfg)
    logger = get_logger()

    benchmark_paths = [pathlib.Path(p) for p in args.benchmarks]
    for path in benchmark_paths:
        if not path.exists():
            sys.exit(f"Benchmark file not found: {path}")

    artifacts = prepare_artifacts(cfg, args.index_prefix)

    # Namespace mimicking CLI args expected by get_answer
    runtime_args = SimpleNamespace(
        mode="chat",
        index_prefix=args.index_prefix,
        model_path=args.model_path,
        system_prompt_mode=args.system_prompt_mode or cfg.system_prompt_mode,
    )

    total_runs = max(1, args.runs)
    for bench_path in benchmark_paths:
        benchmarks = load_benchmarks(bench_path)
        print(f"Loaded {len(benchmarks)} benchmarks from {bench_path}")
        for run_idx in range(1, total_runs + 1):
            if total_runs > 1:
                print(f"\n====== Run {run_idx}/{total_runs} ({bench_path}) ======")
            for bench in benchmarks:
                bench_id = bench.get("id") or bench.get("question", "")[:16]
                question = bench["question"]
                golden_chunks: Optional[List[str]] = bench.get("golden_chunks")

                print(f"\n----- Benchmark: {bench_id} -----")
                print(f"Question: {question}")

                result = get_answer(
                    question=question,
                    cfg=cfg,
                    args=runtime_args,
                    logger=logger,
                    artifacts=artifacts,
                    golden_chunks=golden_chunks,
                    is_test_mode=True,
                )

                if isinstance(result, tuple):
                    answer_text, chunks_info, hyde_query = result
                else:
                    answer_text = result
                    chunks_info = None
                    hyde_query = None

                print("\nAnswer:")
                print(answer_text.strip() if answer_text else "(empty answer)")

                if hyde_query:
                    print("\nHyDE query used:")
                    print(hyde_query.strip())

                if chunks_info:
                    print("\nTop chunks:")
                    for chunk in chunks_info:
                        print(
                            f"  #{chunk['rank']} chunk_id={chunk['chunk_id']} "
                            f"(FAISS rank={chunk['faiss_rank']}, BM25 rank={chunk['bm25_rank']})"
                        )

                logger.log_generation(
                    answer_text,
                    {
                        "max_tokens": cfg.max_gen_tokens,
                        "model_path": runtime_args.model_path or cfg.model_path,
                    },
                )
                logger.log_query_complete()

    print("\nBenchmark run complete. Detailed logs: ", logger.log_file)


if __name__ == "__main__":
    main()
