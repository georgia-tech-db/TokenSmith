"""
profile_retrieval_plans.py

Profiles all retrieval plan combinations on the benchmark suite.
For each query in benchmarks.yaml, runs every plan and records:
  - Retrieval latency (per-retriever and total)
  - Retrieval quality (hit rate against ideal_retrieved_chunks)

Plans tested:
  1. FAISS-only
  2. BM25-only
  3. FAISS+BM25 (equal weight)
  4. FAISS+BM25+Keyword (0.4/0.4/0.2)
  5. Each of the above with cross-encoder reranking

Usage:
  python -m scripts.profile_retrieval_plans                    # run with real models
  python -m scripts.profile_retrieval_plans --simulate         # generate sample data
  python -m scripts.profile_retrieval_plans --output results/  # custom output dir
"""

import argparse
import json
import os
import pathlib
import random
import sys
import time
import yaml
from typing import Any, Dict, List, Tuple

# Ensure project root is on sys.path
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ── Plan definitions ────────────────────────────────────────────────────────

PLANS = {
    "faiss_only": {
        "weights": {"faiss": 1.0, "bm25": 0.0, "index_keywords": 0.0},
        "rerank": False,
        "label": "FAISS Only",
    },
    "bm25_only": {
        "weights": {"faiss": 0.0, "bm25": 1.0, "index_keywords": 0.0},
        "rerank": False,
        "label": "BM25 Only",
    },
    "faiss_bm25": {
        "weights": {"faiss": 0.5, "bm25": 0.5, "index_keywords": 0.0},
        "rerank": False,
        "label": "FAISS + BM25",
    },
    "full_ensemble": {
        "weights": {"faiss": 0.4, "bm25": 0.4, "index_keywords": 0.2},
        "rerank": False,
        "label": "Full Ensemble",
    },
    "faiss_only_rerank": {
        "weights": {"faiss": 1.0, "bm25": 0.0, "index_keywords": 0.0},
        "rerank": True,
        "label": "FAISS Only + Rerank",
    },
    "bm25_only_rerank": {
        "weights": {"faiss": 0.0, "bm25": 1.0, "index_keywords": 0.0},
        "rerank": True,
        "label": "BM25 Only + Rerank",
    },
    "faiss_bm25_rerank": {
        "weights": {"faiss": 0.5, "bm25": 0.5, "index_keywords": 0.0},
        "rerank": True,
        "label": "FAISS + BM25 + Rerank",
    },
    "full_ensemble_rerank": {
        "weights": {"faiss": 0.4, "bm25": 0.4, "index_keywords": 0.2},
        "rerank": True,
        "label": "Full Ensemble + Rerank",
    },
}


# ── Retrieval quality metric ───────────────────────────────────────────────

def hit_rate_at_k(retrieved_ids: List[int], ideal_ids: List[int], k: int = 10) -> float:
    """Fraction of ideal chunks found in the top-k retrieved chunks."""
    if not ideal_ids:
        return 0.0
    retrieved_set = set(retrieved_ids[:k])
    hits = sum(1 for cid in ideal_ids if cid in retrieved_set)
    return hits / len(ideal_ids)


def ndcg_at_k(retrieved_ids: List[int], ideal_ids: List[int], k: int = 10) -> float:
    """Normalized discounted cumulative gain at k."""
    import math
    ideal_set = set(ideal_ids)
    # DCG of retrieved list
    dcg = 0.0
    for i, rid in enumerate(retrieved_ids[:k]):
        if rid in ideal_set:
            dcg += 1.0 / math.log2(i + 2)  # i+2 because rank starts at 1
    # Ideal DCG (all ideal chunks at top positions)
    idcg = 0.0
    for i in range(min(len(ideal_ids), k)):
        idcg += 1.0 / math.log2(i + 2)
    return dcg / idcg if idcg > 0 else 0.0


# ── Real profiling (requires models + index) ──────────────────────────────

def run_real_profiling(benchmarks: List[Dict], config_path: str, output_dir: pathlib.Path):
    """Run actual retrieval with each plan and measure latency + quality."""
    from src.config import RAGConfig
    from src.retriever import (
        FAISSRetriever, BM25Retriever, IndexKeywordRetriever,
        load_artifacts, filter_retrieved_chunks
    )
    from src.ranking.ranker import EnsembleRanker
    from src.ranking.reranker import rerank

    cfg = RAGConfig.from_yaml(config_path)
    artifacts_dir = cfg.get_artifacts_directory()
    faiss_idx, bm25_idx, chunks, sources, meta = load_artifacts(artifacts_dir, "textbook_index")

    results = []

    for bench in benchmarks:
        qid = bench["id"]
        question = bench["question"]
        ideal_chunks = bench.get("ideal_retrieved_chunks", [])

        print(f"\n  Profiling: {qid} — {question[:60]}...")

        for plan_name, plan_cfg in PLANS.items():
            weights = plan_cfg["weights"]
            do_rerank = plan_cfg["rerank"]

            # Build retrievers based on plan weights
            retrievers = []
            active_weights = {}
            if weights["faiss"] > 0:
                retrievers.append(FAISSRetriever(faiss_idx, cfg.embed_model))
                active_weights["faiss"] = weights["faiss"]
            if weights["bm25"] > 0:
                retrievers.append(BM25Retriever(bm25_idx))
                active_weights["bm25"] = weights["bm25"]
            if weights["index_keywords"] > 0:
                retrievers.append(IndexKeywordRetriever(
                    cfg.extracted_index_path, cfg.page_to_chunk_map_path
                ))
                active_weights["index_keywords"] = weights["index_keywords"]

            # Normalize weights
            total_w = sum(active_weights.values())
            active_weights = {k: v / total_w for k, v in active_weights.items()}

            ranker = EnsembleRanker(
                ensemble_method="rrf",
                weights=active_weights,
                rrf_k=60,
            )

            pool_n = max(cfg.num_candidates, cfg.top_k + 10)

            # ── Timed retrieval ──
            t_start = time.perf_counter()

            raw_scores: Dict[str, Dict[int, float]] = {}
            retriever_timings = {}
            for retriever in retrievers:
                t_r = time.perf_counter()
                raw_scores[retriever.name] = retriever.get_scores(question, pool_n, chunks)
                retriever_timings[retriever.name] = (time.perf_counter() - t_r) * 1000

            t_rank = time.perf_counter()
            ordered, scores = ranker.rank(raw_scores=raw_scores)
            topk_idxs = filter_retrieved_chunks(cfg, chunks, ordered)
            rank_ms = (time.perf_counter() - t_rank) * 1000

            rerank_ms = 0.0
            if do_rerank:
                ranked_chunks = [chunks[i] for i in topk_idxs]
                t_rr = time.perf_counter()
                rerank(question, ranked_chunks, mode="cross_encoder", top_n=cfg.rerank_top_k)
                rerank_ms = (time.perf_counter() - t_rr) * 1000

            total_ms = (time.perf_counter() - t_start) * 1000

            # ── Quality ──
            hr = hit_rate_at_k(topk_idxs, ideal_chunks, k=cfg.top_k)
            ndcg = ndcg_at_k(topk_idxs, ideal_chunks, k=cfg.top_k)

            result = {
                "query_id": qid,
                "question": question,
                "plan": plan_name,
                "plan_label": plan_cfg["label"],
                "weights": weights,
                "rerank": do_rerank,
                "latency_ms": round(total_ms, 2),
                "retriever_latency_ms": {k: round(v, 2) for k, v in retriever_timings.items()},
                "rank_latency_ms": round(rank_ms, 2),
                "rerank_latency_ms": round(rerank_ms, 2),
                "hit_rate_at_10": round(hr, 4),
                "ndcg_at_10": round(ndcg, 4),
                "top_k_ids": topk_idxs[:10],
                "ideal_chunk_ids": ideal_chunks,
            }
            results.append(result)
            print(f"    {plan_cfg['label']:30s}  {total_ms:7.1f}ms  HR@10={hr:.3f}  NDCG@10={ndcg:.3f}")

    return results


# ── Simulated profiling (no models needed) ────────────────────────────────

def run_simulated_profiling(benchmarks: List[Dict]) -> List[Dict]:
    """
    Generate realistic simulated profiling data based on known retrieval
    characteristics. Used for development and chart generation when models
    are unavailable.

    Latency estimates are based on published FAISS/BM25 benchmarks for
    ~1500 chunks on Apple M-series CPU:
      - FAISS (FlatL2, 384-dim, 1500 vectors): ~15-40ms
      - BM25 (rank_bm25, 1500 docs):           ~5-15ms
      - IndexKeyword (JSON lookup):             ~1-5ms
      - Cross-encoder reranking (5 pairs):      ~80-200ms
      - RRF fusion:                             ~1-3ms
    """
    random.seed(42)
    results = []

    # Latency profiles per retriever (mean, stddev) in ms
    LATENCY = {
        "faiss":          (28.0, 8.0),
        "bm25":           (10.0, 3.0),
        "index_keywords": (3.0,  1.0),
        "rank_fusion":    (2.0,  0.5),
        "rerank":         (140.0, 30.0),
    }

    # Quality profiles: how well each plan type does on different query types
    # query_type -> plan -> (hit_rate_mean, ndcg_mean)
    QUERY_TYPES = {
        "keyword_heavy": ["acid_properties", "primary_foreign_keys", "fd_normalization",
                          "sql_isolation", "book_authors", "database_schema"],
        "semantic":      ["oltp_vs_analytics", "lossy_decomposition", "aggregation_grouping",
                          "aries_atomicity"],
        "factual":       ["bptree"],
    }

    def classify_query(qid):
        for qtype, ids in QUERY_TYPES.items():
            if qid in ids:
                return qtype
        return "semantic"

    # Base quality by plan and query type: (hit_rate, ndcg)
    QUALITY = {
        "keyword_heavy": {
            "faiss_only": (0.52, 0.45), "bm25_only": (0.68, 0.60),
            "faiss_bm25": (0.72, 0.65), "full_ensemble": (0.74, 0.67),
        },
        "semantic": {
            "faiss_only": (0.70, 0.63), "bm25_only": (0.38, 0.30),
            "faiss_bm25": (0.72, 0.65), "full_ensemble": (0.73, 0.66),
        },
        "factual": {
            "faiss_only": (0.60, 0.52), "bm25_only": (0.62, 0.55),
            "faiss_bm25": (0.66, 0.58), "full_ensemble": (0.68, 0.60),
        },
    }

    for bench in benchmarks:
        qid = bench["id"]
        question = bench["question"]
        ideal_chunks = bench.get("ideal_retrieved_chunks", [])
        qtype = classify_query(qid)

        for plan_name, plan_cfg in PLANS.items():
            weights = plan_cfg["weights"]
            do_rerank = plan_cfg["rerank"]
            base_plan = plan_name.replace("_rerank", "")

            # Compute latency
            retriever_latency = {}
            for rname in ["faiss", "bm25", "index_keywords"]:
                if weights[rname] > 0:
                    mean, std = LATENCY[rname]
                    retriever_latency[rname] = max(1.0, random.gauss(mean, std))

            rank_ms = max(0.5, random.gauss(*LATENCY["rank_fusion"]))
            rerank_ms = max(50.0, random.gauss(*LATENCY["rerank"])) if do_rerank else 0.0
            total_ms = sum(retriever_latency.values()) + rank_ms + rerank_ms

            # Compute quality
            base_hr, base_ndcg = QUALITY[qtype].get(base_plan, (0.5, 0.4))
            noise = random.gauss(0, 0.04)
            hr = max(0.0, min(1.0, base_hr + noise))
            ndcg = max(0.0, min(1.0, base_ndcg + noise))

            # Reranking boosts quality slightly
            if do_rerank:
                hr = min(1.0, hr + random.uniform(0.02, 0.08))
                ndcg = min(1.0, ndcg + random.uniform(0.03, 0.10))

            # Simulate retrieved chunk IDs
            n_ideal = len(ideal_chunks)
            n_hits = int(hr * n_ideal)
            simulated_topk = ideal_chunks[:n_hits] + [
                random.randint(0, 1500) for _ in range(10 - n_hits)
            ]

            result = {
                "query_id": qid,
                "question": question,
                "query_type": qtype,
                "plan": plan_name,
                "plan_label": plan_cfg["label"],
                "weights": weights,
                "rerank": do_rerank,
                "latency_ms": round(total_ms, 2),
                "retriever_latency_ms": {k: round(v, 2) for k, v in retriever_latency.items()},
                "rank_latency_ms": round(rank_ms, 2),
                "rerank_latency_ms": round(rerank_ms, 2),
                "hit_rate_at_10": round(hr, 4),
                "ndcg_at_10": round(ndcg, 4),
                "top_k_ids": simulated_topk[:10],
                "ideal_chunk_ids": ideal_chunks,
            }
            results.append(result)

    return results


# ── Summary statistics ─────────────────────────────────────────────────────

def compute_summary(results: List[Dict]) -> Dict[str, Any]:
    """Aggregate per-plan statistics across all queries."""
    from collections import defaultdict

    plan_stats = defaultdict(lambda: {
        "latencies": [], "hit_rates": [], "ndcg_scores": [],
        "retriever_latencies": defaultdict(list),
    })

    for r in results:
        p = r["plan"]
        plan_stats[p]["latencies"].append(r["latency_ms"])
        plan_stats[p]["hit_rates"].append(r["hit_rate_at_10"])
        plan_stats[p]["ndcg_scores"].append(r["ndcg_at_10"])
        for rname, lat in r["retriever_latency_ms"].items():
            plan_stats[p]["retriever_latencies"][rname].append(lat)

    summary = {}
    for plan_name, stats in plan_stats.items():
        lats = stats["latencies"]
        hrs = stats["hit_rates"]
        ndcgs = stats["ndcg_scores"]
        summary[plan_name] = {
            "label": PLANS[plan_name]["label"],
            "avg_latency_ms": round(sum(lats) / len(lats), 2),
            "p50_latency_ms": round(sorted(lats)[len(lats) // 2], 2),
            "p95_latency_ms": round(sorted(lats)[int(len(lats) * 0.95)], 2),
            "avg_hit_rate": round(sum(hrs) / len(hrs), 4),
            "avg_ndcg": round(sum(ndcgs) / len(ndcgs), 4),
            "n_queries": len(lats),
        }

    return summary


def compute_per_query_type_summary(results: List[Dict]) -> Dict[str, Dict]:
    """Aggregate stats grouped by query type and plan."""
    from collections import defaultdict

    grouped = defaultdict(lambda: defaultdict(lambda: {"latencies": [], "hit_rates": [], "ndcg_scores": []}))

    for r in results:
        qtype = r.get("query_type", "unknown")
        plan = r["plan"]
        grouped[qtype][plan]["latencies"].append(r["latency_ms"])
        grouped[qtype][plan]["hit_rates"].append(r["hit_rate_at_10"])
        grouped[qtype][plan]["ndcg_scores"].append(r["ndcg_at_10"])

    summary = {}
    for qtype, plans in grouped.items():
        summary[qtype] = {}
        for plan_name, stats in plans.items():
            lats = stats["latencies"]
            hrs = stats["hit_rates"]
            ndcgs = stats["ndcg_scores"]
            summary[qtype][plan_name] = {
                "label": PLANS[plan_name]["label"],
                "avg_latency_ms": round(sum(lats) / len(lats), 2),
                "avg_hit_rate": round(sum(hrs) / len(hrs), 4),
                "avg_ndcg": round(sum(ndcgs) / len(ndcgs), 4),
            }

    return summary


def build_lookup_table(per_type_summary: Dict) -> Dict[str, str]:
    """
    For each query type, select the plan that minimizes latency while
    keeping hit_rate within 5% of the best plan's hit_rate.

    Returns: {query_type: recommended_plan_name}
    """
    lookup = {}

    for qtype, plans in per_type_summary.items():
        # Find best quality (hit_rate) across all plans for this query type
        best_hr = max(p["avg_hit_rate"] for p in plans.values())
        quality_threshold = best_hr * 0.95  # within 5%

        # Among plans meeting quality threshold, pick lowest latency
        eligible = [
            (pname, pstats) for pname, pstats in plans.items()
            if pstats["avg_hit_rate"] >= quality_threshold
        ]

        if eligible:
            best_plan = min(eligible, key=lambda x: x[1]["avg_latency_ms"])
            lookup[qtype] = best_plan[0]
        else:
            # Fallback: pick highest quality
            best_plan = max(plans.items(), key=lambda x: x[1]["avg_hit_rate"])
            lookup[qtype] = best_plan[0]

    return lookup


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Profile retrieval plan combinations")
    parser.add_argument("--simulate", action="store_true",
                        help="Generate simulated profiling data (no models needed)")
    parser.add_argument("--config", default="config/config.yaml",
                        help="Path to config.yaml")
    parser.add_argument("--benchmarks", default="tests/benchmarks.yaml",
                        help="Path to benchmarks.yaml")
    parser.add_argument("--output", default="results/profiling",
                        help="Output directory for results")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)

    # Load benchmarks
    with open(args.benchmarks, "r") as f:
        data = yaml.safe_load(f)
    benchmarks = data["benchmarks"]
    print(f"Loaded {len(benchmarks)} benchmark queries.")

    # Create output directory
    output_dir = pathlib.Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run profiling
    if args.simulate:
        print("Running SIMULATED profiling (no models required)...")
        results = run_simulated_profiling(benchmarks)
    else:
        print("Running REAL profiling (requires models + index)...")
        results = run_real_profiling(benchmarks, args.config, output_dir)

    # Save raw results
    raw_path = output_dir / "profiling_results.json"
    with open(raw_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} profiling results to {raw_path}")

    # Compute and save summary
    summary = compute_summary(results)
    summary_path = output_dir / "plan_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Compute per-query-type summary and lookup table
    per_type = compute_per_query_type_summary(results)
    per_type_path = output_dir / "per_query_type_summary.json"
    with open(per_type_path, "w") as f:
        json.dump(per_type, f, indent=2)

    lookup = build_lookup_table(per_type)
    lookup_path = output_dir / "lookup_table.json"
    with open(lookup_path, "w") as f:
        json.dump(lookup, f, indent=2)

    # Print summary table
    print(f"\n{'='*80}")
    print(f"  {'Plan':<30s}  {'Avg Latency':>12s}  {'P50':>8s}  {'P95':>8s}  {'HR@10':>8s}  {'NDCG@10':>8s}")
    print(f"{'='*80}")
    for plan_name in PLANS:
        s = summary[plan_name]
        print(f"  {s['label']:<30s}  {s['avg_latency_ms']:>10.1f}ms  {s['p50_latency_ms']:>6.1f}ms"
              f"  {s['p95_latency_ms']:>6.1f}ms  {s['avg_hit_rate']:>7.4f}  {s['avg_ndcg']:>7.4f}")
    print(f"{'='*80}")

    # Print lookup table
    print(f"\n  Recommended plans (static lookup table):")
    print(f"  {'Query Type':<20s}  {'Recommended Plan':<30s}")
    print(f"  {'-'*50}")
    for qtype, plan in lookup.items():
        print(f"  {qtype:<20s}  {PLANS[plan]['label']:<30s}")
    print()


if __name__ == "__main__":
    main()
