#!/usr/bin/env python3
"""
Evaluate retrieval-context caching against a no-cache baseline.

The experiment runs each case in two modes:
1) no_cache: normal pipeline, no retrieval context cache
2) context_chunk_cache: retrieval context cache enabled, generation still fresh

Outputs JSON summary with:
- retrieval runtime profile
- end-to-end runtime
- retrieved chunk-id equality checks
"""

from __future__ import annotations

import argparse
import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import yaml

from src.config import RAGConfig
from src.instrumentation.logging import get_logger
from src.main import get_answer
from src.ranking.ranker import EnsembleRanker
from src.retriever import BM25Retriever, FAISSRetriever, IndexKeywordRetriever, load_artifacts


def load_corpus(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("cases", [])


def build_artifacts(cfg: RAGConfig, index_prefix: str) -> Dict[str, Any]:
    artifacts_dir = cfg.get_artifacts_directory()
    faiss_index, bm25_index, chunks, sources, metadata = load_artifacts(
        artifacts_dir=artifacts_dir, index_prefix=index_prefix
    )
    retrievers = [FAISSRetriever(faiss_index, cfg.embed_model), BM25Retriever(bm25_index)]
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
        "meta": metadata,
    }


def run_query(
    question: str,
    prompt_mode: str,
    cfg: RAGConfig,
    artifacts: Dict[str, Any],
    index_prefix: str,
) -> Dict[str, Any]:
    import argparse as _argparse

    args = _argparse.Namespace(
        index_prefix=index_prefix,
        model_path=cfg.gen_model,
        system_prompt_mode=prompt_mode,
        double_prompt=False,
    )
    logger = get_logger()
    additional_log_info: Dict[str, Any] = {}
    start = time.perf_counter()
    result = get_answer(
        question=question,
        cfg=cfg,
        args=args,
        logger=logger,
        console=None,
        artifacts=artifacts,
        is_test_mode=True,
        additional_log_info=additional_log_info,
    )
    total_runtime_ms = round((time.perf_counter() - start) * 1000.0, 3)
    if isinstance(result, tuple):
        _, chunks_info, _ = result
    else:
        chunks_info = []
    chunk_ids = [int(item["chunk_id"]) for item in (chunks_info or []) if "chunk_id" in item]
    return {
        "chunk_ids": chunk_ids,
        "total_runtime_ms": total_runtime_ms,
        "retrieval_profile": additional_log_info.get("retrieval_profile", {}),
    }


def run_case(case: Dict[str, Any], cfg: RAGConfig, artifacts: Dict[str, Any], index_prefix: str) -> Dict[str, Any]:
    question = case["question"]
    first_mode = case["first_prompt_mode"]
    second_mode = case["second_prompt_mode"]

    # 1) No cache baseline
    no_cache_cfg = deepcopy(cfg)
    no_cache_cfg.enable_context_chunk_cache = False
    no_cache_first = run_query(question, first_mode, no_cache_cfg, artifacts, index_prefix)
    no_cache_second = run_query(question, second_mode, no_cache_cfg, artifacts, index_prefix)

    # 2) Retrieval context cache mode
    context_cfg = deepcopy(cfg)
    context_cfg.enable_context_chunk_cache = True
    context_first = run_query(question, first_mode, context_cfg, artifacts, index_prefix)
    context_second = run_query(question, second_mode, context_cfg, artifacts, index_prefix)
    context_cache_hit = bool(context_second["retrieval_profile"].get("cache_hit", False))
    chunk_ids_match = context_second["chunk_ids"] == no_cache_second["chunk_ids"]
    context_retrieval_ms = float(context_second["retrieval_profile"].get("retrieval_ms", 0.0))
    no_cache_retrieval_ms = float(no_cache_second["retrieval_profile"].get("retrieval_ms", 0.0))
    retrieval_speedup_ms = round(no_cache_retrieval_ms - context_retrieval_ms, 3)

    return {
        "id": case["id"],
        "question": question,
        "first_prompt_mode": first_mode,
        "second_prompt_mode": second_mode,
        "checks": {
            "context_cache_hit_on_second_run": context_cache_hit,
            "cached_chunks_match_no_cache_chunks": chunk_ids_match,
            "cached_chunks_match_first_context_run": context_second["chunk_ids"] == context_first["chunk_ids"],
            "no_cache_chunks_stable_across_runs": no_cache_first["chunk_ids"] == no_cache_second["chunk_ids"],
        },
        "summary": {
            "no_cache_retrieval_ms": no_cache_retrieval_ms,
            "context_cache_retrieval_ms": context_retrieval_ms,
            "retrieval_speedup_ms": retrieval_speedup_ms,
            "no_cache_total_runtime_ms": no_cache_second["total_runtime_ms"],
            "context_cache_total_runtime_ms": context_second["total_runtime_ms"],
        },
        "no_cache_second": {
            "runtime_ms": no_cache_second["total_runtime_ms"],
            "retrieval_profile": no_cache_second["retrieval_profile"],
            "chunk_ids": no_cache_second["chunk_ids"],
        },
        "context_chunk_cache_second": {
            "runtime_ms": context_second["total_runtime_ms"],
            "retrieval_profile": context_second["retrieval_profile"],
            "chunk_ids": context_second["chunk_ids"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run context-cache evaluation corpus")
    parser.add_argument("--corpus", default="tests/cache_context_corpus.yaml")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--index-prefix", default="textbook_index")
    parser.add_argument("--output", default="tests/results/cache_context_experiment.json")
    parser.add_argument(
        "--model-path",
        default=None,
        help="Optional override for generation model path (.gguf).",
    )
    parser.add_argument(
        "--max-gen-tokens",
        type=int,
        default=None,
        help="Optional override for max generation tokens.",
    )
    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = RAGConfig.from_yaml(args.config)
    if args.max_gen_tokens is not None:
        cfg.max_gen_tokens = int(args.max_gen_tokens)
    if args.model_path:
        cfg.gen_model = args.model_path
    if not Path(cfg.gen_model).exists():
        fallback_models = sorted(Path("models").glob("*.gguf"))
        preferred = [
            m for m in fallback_models
            if "instruct" in m.name.lower() or "chat" in m.name.lower()
        ]
        fallback_models = preferred or fallback_models
        if fallback_models:
            cfg.gen_model = str(fallback_models[0])
            print(f"[cache-experiment] Configured model not found. Using fallback: {cfg.gen_model}")
        else:
            raise FileNotFoundError(
                f"Generation model not found: {cfg.gen_model}. "
                "Pass --model-path <path_to_model.gguf> or add a .gguf under models/."
            )

    cases = load_corpus(corpus_path)
    print(
        f"[cache-experiment] Loaded {len(cases)} cases | model={cfg.gen_model} | max_gen_tokens={cfg.max_gen_tokens}",
        flush=True,
    )
    artifacts = build_artifacts(cfg, args.index_prefix)
    print(
        f"[cache-experiment] Artifacts ready: chunks={len(artifacts['chunks'])}, retrievers={len(artifacts['retrievers'])}",
        flush=True,
    )

    results = []
    for idx, case in enumerate(cases, 1):
        print(f"[cache-experiment] Running case {idx}/{len(cases)}: {case.get('id', 'unknown')}", flush=True)
        start = time.perf_counter()
        case_result = run_case(case, cfg, artifacts, args.index_prefix)
        elapsed = round((time.perf_counter() - start), 2)
        print(f"[cache-experiment] Finished case {idx}/{len(cases)} in {elapsed}s", flush=True)
        results.append(case_result)
    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "corpus": str(corpus_path),
        "cases": results,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Saved cache-context experiment results to: {out_path}")


if __name__ == "__main__":
    main()
