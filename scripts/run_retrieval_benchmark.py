#!/usr/bin/env python3
"""Run retrieval-only TokenSmith benchmarks and score them with ranked IR metrics."""

from __future__ import annotations

import argparse
import importlib
import json
import pathlib
import sys
import time
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional

import yaml

from tests.metrics import SimilarityScorer


RETRIEVAL_METRICS = [
    "chunk_ndcg_10",
    "chunk_recall_5",
    "chunk_recall_10",
    "chunk_mrr_10",
    "chunk_map_10",
    "page_hit_5",
    "page_hit_10",
    "direct_page_hit_10",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TokenSmith retrieval-only benchmarks.")
    parser.add_argument("--repo-root", default=".", help="Target repository root.")
    parser.add_argument("--config", default="config/config.yaml", help="Config path relative to repo root.")
    parser.add_argument("--benchmarks", default="tests/benchmarks.yaml", help="Benchmark yaml path.")
    parser.add_argument("--artifacts-dir", default="index/sections", help="Artifacts directory relative to repo root.")
    parser.add_argument("--index-prefix", default="textbook_index", help="Artifact prefix.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument(
        "--mode",
        choices=["auto", "improved", "baseline"],
        default="auto",
        help="Force retrieval mode. 'auto' detects from repo contents. "
        "'baseline' runs the legacy ensemble-ranker path (useful for running "
        "baseline benchmarks from the improved branch against old artifacts).",
    )
    return parser.parse_args()


def _load_benchmarks(path: pathlib.Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return list(data.get("benchmarks", []))


def _composite_retrieval_score(scores: Dict[str, Any]) -> float:
    values = [float(scores.get(f"{metric}_similarity", 0.0)) for metric in RETRIEVAL_METRICS]
    return mean(values) if values else 0.0


def _build_chunk_records(
    chunk_ids: Iterable[int],
    chunks: List[str],
    metadata: List[Dict[str, Any]],
    ordered_scores: Optional[Iterable[float]] = None,
) -> List[Dict[str, Any]]:
    scores = list(ordered_scores or [])
    records: List[Dict[str, Any]] = []
    for rank, chunk_id in enumerate(chunk_ids, start=1):
        score = float(scores[rank - 1]) if rank - 1 < len(scores) else 0.0
        meta = metadata[int(chunk_id)]
        records.append(
            {
                "rank": rank,
                "chunk_id": int(chunk_id),
                "content": chunks[int(chunk_id)],
                "page_numbers": [int(page) for page in meta.get("page_numbers", [])],
                "section_path": meta.get("section_path"),
                "score": score,
            }
        )
    return records


def _prepare_improved(
    repo_root: pathlib.Path,
    config_path: pathlib.Path,
    artifacts_dir: pathlib.Path,
    index_prefix: str,
) -> Dict[str, Any]:
    sys.path.insert(0, str(repo_root))
    config_mod = importlib.import_module("src.config")
    retriever_mod = importlib.import_module("src.retriever")
    pipeline_mod = importlib.import_module("src.retrieval_pipeline")

    cfg = config_mod.RAGConfig.from_yaml(config_path)
    bundle = retriever_mod.load_artifact_bundle(artifacts_dir, index_prefix)
    runtime_retrievers = pipeline_mod.build_runtime_retrievers(bundle, cfg)
    return {
        "cfg": cfg,
        "bundle": bundle,
        "runtime_retrievers": runtime_retrievers,
        "pipeline_mod": pipeline_mod,
    }


def _run_improved(prepared: Dict[str, Any], benchmark: Dict[str, Any]) -> Dict[str, Any]:
    cfg = prepared["cfg"]
    bundle = prepared["bundle"]
    runtime_retrievers = prepared["runtime_retrievers"]
    pipeline_mod = prepared["pipeline_mod"]

    start = time.perf_counter()
    _ranked_chunks, chunk_ids, trace = pipeline_mod.execute_retrieval_plan(
        query=benchmark["question"],
        cfg=cfg,
        bundle=bundle,
        retrievers=runtime_retrievers,
        history=benchmark.get("history", []),
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    chunk_scores = list(trace.fused_chunk_scores)
    chunks_info = _build_chunk_records(chunk_ids, bundle.chunks, bundle.metadata, chunk_scores)
    return {
        "chunks_info": chunks_info,
        "retrieval_trace": {
            **trace.__dict__,
            "retrieval_latency_ms": float(trace.retrieval_latency_ms or elapsed_ms),
            "total_latency_ms": float(trace.total_latency_ms or elapsed_ms),
        },
    }


def _prepare_baseline(
    repo_root: pathlib.Path,
    config_path: pathlib.Path,
    artifacts_dir: pathlib.Path,
    index_prefix: str,
) -> Dict[str, Any]:
    # Always use the current branch's modules (with hardened embedder) even when
    # scoring baseline artifacts, so the embedding model loads reliably.
    from src.config import RAGConfig as config_cls
    from src.retriever import (
        BM25Retriever,
        FAISSRetriever,
        IndexKeywordRetriever,
        filter_retrieved_chunks,
        load_artifacts,
    )
    from src.ranking.ranker import EnsembleRanker

    cfg = config_cls.from_yaml(config_path)
    index, bm25_index, chunks, _sources, metadata = load_artifacts(artifacts_dir, index_prefix)

    retrievers = [
        FAISSRetriever(index, cfg.embed_model),
        BM25Retriever(bm25_index),
    ]
    if cfg.ranker_weights.get("index_keywords", 0) > 0:
        retrievers.append(
            IndexKeywordRetriever(
                cfg.extracted_index_path,
                cfg.page_to_chunk_map_path,
            )
        )
    ranker = EnsembleRanker(
        ensemble_method=cfg.ensemble_method,
        weights=cfg.ranker_weights,
        rrf_k=int(cfg.rrf_k),
    )
    return {
        "cfg": cfg,
        "chunks": chunks,
        "metadata": metadata,
        "retrievers": retrievers,
        "ranker": ranker,
        "filter_fn": filter_retrieved_chunks,
    }


def _run_baseline(prepared: Dict[str, Any], benchmark: Dict[str, Any]) -> Dict[str, Any]:
    cfg = prepared["cfg"]
    chunks = prepared["chunks"]
    metadata = prepared["metadata"]
    retrievers = prepared["retrievers"]
    ranker = prepared["ranker"]
    filter_fn = prepared["filter_fn"]

    start = time.perf_counter()
    raw_scores: Dict[str, Dict[int, float]] = {}
    for retriever in retrievers:
        raw_scores[retriever.name] = retriever.get_scores(benchmark["question"], cfg.num_candidates, chunks)
    ordered, ordered_scores = ranker.rank(raw_scores=raw_scores)
    chunk_ids = filter_fn(cfg, chunks, ordered)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    chunks_info = _build_chunk_records(chunk_ids, chunks, metadata, ordered_scores[: len(chunk_ids)])
    return {
        "chunks_info": chunks_info,
        "retrieval_trace": {
            "query_type": None,
            "resolved_query_type": None,
            "retrieval_mode": "flat",
            "route_reason": "baseline",
            "chunk_scores": raw_scores,
            "fused_chunk_ids": [int(cid) for cid in ordered[: len(chunk_ids)]],
            "fused_chunk_scores": [float(score) for score in ordered_scores[: len(chunk_ids)]],
            "retrieval_latency_ms": elapsed_ms,
            "total_latency_ms": elapsed_ms,
            "chunks_passed_to_generation": len(chunk_ids),
            "prompt_tokens_estimate": 0,
            "selected_section_paths": [],
            "subquery_traces": [],
        },
    }


def main() -> None:
    args = parse_args()
    repo_root = pathlib.Path(args.repo_root).resolve()
    config_path = (repo_root / args.config).resolve()
    artifacts_dir = (repo_root / args.artifacts_dir).resolve()
    output_path = pathlib.Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    benchmarks = _load_benchmarks(pathlib.Path(args.benchmarks).resolve())
    scorer = SimilarityScorer(enabled_metrics=RETRIEVAL_METRICS)

    if args.mode == "auto":
        is_improved = (repo_root / "src" / "retrieval_pipeline.py").exists()
    elif args.mode == "improved":
        is_improved = True
    else:
        is_improved = False

    if is_improved:
        prepared = _prepare_improved(repo_root, config_path, artifacts_dir, args.index_prefix)
        runner = _run_improved
    else:
        prepared = _prepare_baseline(repo_root, config_path, artifacts_dir, args.index_prefix)
        runner = _run_baseline

    with output_path.open("w", encoding="utf-8") as handle:
        for benchmark in benchmarks:
            result = runner(prepared, benchmark)
            scores = scorer.calculate_scores(
                answer="",
                expected="",
                keywords=[],
                question=benchmark["question"],
                retrieval_gold=benchmark.get("retrieval_gold"),
                actual_retrieved_chunks=result["chunks_info"],
            )
            scores["final_score"] = _composite_retrieval_score(scores)

            record = {
                "test_id": benchmark["id"],
                "question": benchmark["question"],
                "expected_query_type": benchmark.get("query_type"),
                "retrieval_gold": benchmark.get("retrieval_gold"),
                "chunks_info": result["chunks_info"],
                "retrieval_trace": result["retrieval_trace"],
                "scores": scores,
                "passed": True,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    sys.path.pop(0)
    print(f"Wrote retrieval benchmark results to {output_path}")


if __name__ == "__main__":
    main()
