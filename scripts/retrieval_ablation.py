#!/usr/bin/env python3
"""Run retrieval ablations across FAISS and BM25 weight settings. Example: python scripts/retrieval_ablation.py"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple



# Add project root to sys.path for imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import QueryPlanConfig
from src.ranking.ranker import EnsembleRanker
from src.retriever import BM25Retriever, FAISSRetriever, load_artifacts
from src.instrumentation.logging import init_logger


QUERY = "What is atomicity?"
WEIGHT_PAIRS: Optional[List[Tuple[float, float]]] = [(0.6, 0.4), (0.5, 0.5)]
METHOD = "linear"
TOP_K_OVERRIDE: Optional[int] = None
POOL_SIZE_OVERRIDE: Optional[int] = None
RRF_K = 60
CONFIG_PATH: Optional[Path] = None
OUTPUT_PATH: Optional[Path] = None


def find_config(path: Optional[Path]) -> QueryPlanConfig:
    if path:
        return QueryPlanConfig.from_yaml(path)
    search = [
        Path("~/.config/tokensmith/config.yaml").expanduser(),
        Path("~/.config/tokensmith/config.yml").expanduser(),
        Path("config/config.yaml"),
    ]
    for candidate in search:
        if candidate.exists():
            return QueryPlanConfig.from_yaml(candidate)
    raise FileNotFoundError("Config not provided and no fallback discovered.")


def normalize_pair(faiss_weight: float, bm25_weight: float) -> Dict[str, float]:
    if faiss_weight < 0 or bm25_weight < 0:
        raise ValueError("Weights must be non-negative.")
    total = faiss_weight + bm25_weight
    if total == 0:
        raise ValueError("At least one weight must be positive.")
    faiss_norm = faiss_weight / total
    faiss_norm = float(f"{faiss_norm:.10f}")
    bm25_norm = 1.0 - faiss_norm
    return {"faiss": faiss_norm, "bm25": bm25_norm}


def collect_raw_scores(
    faiss_retriever: FAISSRetriever,
    bm25_retriever: BM25Retriever,
    query: str,
    pool_size: int,
    chunks: List[str],
) -> Dict[str, Dict[int, float]]:
    raw_scores: Dict[str, Dict[int, float]] = {}
    raw_scores["faiss"] = faiss_retriever.get_scores(query, pool_size, chunks)
    raw_scores["bm25"] = bm25_retriever.get_scores(query, pool_size, chunks)
    return raw_scores


def summarize_run(
    ranking: List[int],
    chunks: List[str],
    sources: List[str],
    top_k: int,
) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    for rank, idx in enumerate(ranking[:top_k], start=1):
        if idx < 0 or idx >= len(chunks):
            continue
        preview = (chunks[idx] or "")[:200].replace("\n", " ")
        results.append(
            {
                "chunk_idx": int(idx),
                "rank": rank,
                "source": sources[idx],
                "preview": preview,
            }
        )
    return results


def main() -> None:
    query = QUERY.strip()
    if not query:
        print("No query configured, exiting.")
        return

    cfg = find_config(CONFIG_PATH)
    init_logger(cfg)
    faiss_index, bm25_index, chunks, sources = load_artifacts(cfg)

    pool_size = POOL_SIZE_OVERRIDE or cfg.pool_size
    pool_size = min(pool_size, len(chunks))
    top_k = TOP_K_OVERRIDE or cfg.top_k

    faiss_retriever = FAISSRetriever(faiss_index, cfg.embed_model)
    bm25_retriever = BM25Retriever(bm25_index)
    raw_scores = collect_raw_scores(faiss_retriever, bm25_retriever, query, pool_size, chunks)

    pairs = WEIGHT_PAIRS or [(cfg.ranker_weights.get("faiss", 0.5), cfg.ranker_weights.get("bm25", 0.5))]
    reports = []

    for f_raw, b_raw in pairs:
        weights = normalize_pair(f_raw, b_raw)
        ranker = EnsembleRanker(METHOD, weights, RRF_K)
        ordered = ranker.rank(raw_scores)
        summary = summarize_run(ordered, chunks, sources, top_k)
        reports.append({"weights": weights, "ranking": summary})

        print(f"Weights {weights} -> top {len(summary)} results:")
        for row in summary:
            print(f"  #{row['rank']:02d} idx={row['chunk_idx']} src={row['source']} :: {row['preview']}")
        print()

    if OUTPUT_PATH:
        if OUTPUT_PATH.suffix != ".json":
            raise ValueError("OUTPUT_PATH must end with .json")
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_PATH.write_text(
            json.dumps(
                {
                    "query": query,
                    "method": METHOD,
                    "top_k": top_k,
                    "pool_size": pool_size,
                    "runs": reports,
                },
                indent=2,
            )
        )
        print(f"Saved ablation report to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
