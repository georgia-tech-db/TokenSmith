#!/usr/bin/env python3
"""
Measure startup artifact loading time with and without SQLite catalog metadata.

Usage:
  python scripts/eval_catalog_loading.py --index_prefix textbook_index
  python scripts/eval_catalog_loading.py --index_prefix textbook_index --catalog_db_path data/tokensmith_catalog.db
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

from src.config import RAGConfig
from src.retriever import load_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark artifact load time")
    parser.add_argument("--index_prefix", default="textbook_index")
    parser.add_argument("--catalog_db_path", default=None)
    parser.add_argument("--iterations", type=int, default=5)
    return parser.parse_args()


def run_once(artifacts_dir: Path, index_prefix: str, catalog_db_path: str | None) -> float:
    t0 = time.perf_counter()
    _, _, chunks, _, metadata = load_artifacts(
        artifacts_dir=artifacts_dir,
        index_prefix=index_prefix,
        catalog_db_path=catalog_db_path,
    )
    elapsed = time.perf_counter() - t0
    if len(chunks) != len(metadata):
        raise ValueError("chunks and metadata lengths do not match")
    return elapsed


def main() -> None:
    args = parse_args()
    cfg = RAGConfig.from_yaml("config/config.yaml")
    artifacts_dir = Path(cfg.get_artifacts_directory())
    samples = [
        run_once(artifacts_dir, args.index_prefix, args.catalog_db_path)
        for _ in range(args.iterations)
    ]
    print(f"artifacts_dir={artifacts_dir}")
    print(f"index_prefix={args.index_prefix}")
    print(f"catalog_db_path={args.catalog_db_path}")
    print(f"iterations={args.iterations}")
    print(f"mean_s={statistics.mean(samples):.6f}")
    print(f"median_s={statistics.median(samples):.6f}")
    print(f"min_s={min(samples):.6f}")
    print(f"max_s={max(samples):.6f}")


if __name__ == "__main__":
    main()
