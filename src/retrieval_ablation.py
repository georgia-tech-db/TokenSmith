#!/usr/bin/env python3
"""Run retrieval ablations across FAISS and BM25 weight settings."""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.config import QueryPlanConfig
from src.ranking.ranker import EnsembleRanker
from src.retriever import BM25Retriever, FAISSRetriever, load_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explore retrieval blends for multiple FAISS/BM25 weights."
    )
    parser.add_argument("query", help="Query text to evaluate.")
    parser.add_argument("--config", help="Optional config path overriding defaults.")
    parser.add_argument(
        "--weights",
        nargs="+",
        help="Weight pairs as 'faiss,bm25'. Defaults to config weights.",
    )
    parser.add_argument(
        "--method",
        default="linear",
        choices=["linear", "rrf"],
        help="Fusion method (default: %(default)s).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        help="Number of results to keep per run (defaults to config top_k).",
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        help="Override pool size for retrievers.",
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=60,
        help="RRF constant when method=rrf (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        help="Optional JSON output file summarizing ablation runs.",
    )
    return parser.parse_args()


def find_config(path: Optional[str]) -> QueryPlanConfig:
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


def parse_weight_pairs(raw_pairs: Optional[List[str]], cfg: QueryPlanConfig) -> List[Tuple[float, float]]:
    pairs: List[Tuple[float, float]] = []
    if not raw_pairs:
        w_f = cfg.ranker_weights.get("faiss", 0.5)
        w_b = cfg.ranker_weights.get("bm25", 0.5)
        pairs.append((float(w_f), float(w_b)))
        return pairs
    for raw in raw_pairs:
        try:
            faiss_val, bm25_val = raw.split(",")
            f = float(faiss_val.strip())
            b = float(bm25_val.strip())
        except ValueError as exc:  # pragma: no cover
            raise ValueError(f"Failed to parse weight pair '{raw}'; expected 'faiss,bm25'.") from exc
        if f < 0 or b < 0:
            raise ValueError("Weights must be non-negative.")
        if f == 0 and b == 0:
            raise ValueError("At least one weight must be positive.")
        pairs.append((f, b))
    return pairs


def normalize_pair(faiss_weight: float, bm25_weight: float) -> Dict[str, float]:
    total = faiss_weight + bm25_weight
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
    args = parse_args()
    cfg = find_config(args.config)

    faiss_index, bm25_index, chunks, sources = load_artifacts(cfg)

    pool_size = args.pool_size or cfg.pool_size
    pool_size = min(pool_size, len(chunks))
    top_k = args.top_k or cfg.top_k

    faiss_retriever = FAISSRetriever(faiss_index, cfg.embed_model)
    bm25_retriever = BM25Retriever(bm25_index)
    raw_scores = collect_raw_scores(faiss_retriever, bm25_retriever, args.query, pool_size, chunks)

    pairs = parse_weight_pairs(args.weights, cfg)
    reports = []

    for f_raw, b_raw in pairs:
        weights = normalize_pair(f_raw, b_raw)
        ranker = EnsembleRanker(args.method, weights, args.rrf_k)
        ordered = ranker.rank(raw_scores)
        summary = summarize_run(ordered, chunks, sources, top_k)
        reports.append({"weights": weights, "ranking": summary})

        print(f"Weights {weights} -> top {len(summary)} results:")
        for row in summary:
            print(f"  #{row['rank']:02d} idx={row['chunk_idx']} src={row['source']} :: {row['preview']}")
        print()

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "query": args.query,
                    "method": args.method,
                    "top_k": top_k,
                    "pool_size": pool_size,
                    "runs": reports,
                },
                indent=2,
            )
        )
        print(f"Saved ablation report to {output_path}")


if __name__ == "__main__":
    main()
