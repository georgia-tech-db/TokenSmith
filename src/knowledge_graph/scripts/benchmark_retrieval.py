import argparse
import json
import logging
import os

import yaml
from dotenv import load_dotenv

from src.knowledge_graph.analysis import analyze_query
from src.knowledge_graph.build import RUNS_DIR
from src.knowledge_graph.io import (
    load_canonicalization_data,
    load_graph_chunks_and_tree,
    resolve_run_dir,
)
from src.knowledge_graph.openrouter_client import OpenRouterClient
from src.knowledge_graph.query import CanonicalLookup, KGNodeRetriever
from src.knowledge_graph.utils.prompts import GRADE_PROMPT

logger = logging.getLogger(__name__)


def _grade_with_llm(
    client: OpenRouterClient,
    model: str,
    query: str,
    retrieved: list[tuple[int, str, float]],
) -> list[dict]:
    """Grade a list of (chunk_id, text, score) tuples for relevance to query.

    Returns a list of {"chunk_id": int, "score": 0|1|2, "reason": str} dicts
    in the same order as *retrieved*. score=-1 if the LLM omitted that passage.
    """
    passages = "\n\n".join(
        f"[{i + 1}] {text[:600].strip()}"
        for i, (_, text, _) in enumerate(retrieved)
    )
    prompt = GRADE_PROMPT.format(query=query, passages=passages)
    raw = client.chat(
        model,
        [{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    grades = json.loads(raw).get("grades", [])

    results = []
    for i, (chunk_id, _, _) in enumerate(retrieved):
        grade = next((g for g in grades if g.get("id") == i + 1), {})
        results.append(
            {
                "chunk_id": chunk_id,
                "score": int(grade["score"]) if "score" in grade else -1,
                "reason": grade.get("reason", ""),
            }
        )
    return results


def _ideal_metrics(retrieved_ids: list[int], ideal: set[int], top_k: int) -> dict:
    hits = set(retrieved_ids) & ideal
    return {
        "precision_at_k": len(hits) / top_k if top_k > 0 else 0.0,
        "recall_at_k": len(hits) / len(ideal) if ideal else 0.0,
        "hits": sorted(hits),
    }


def _llm_metrics(grades: list[dict], top_k: int) -> dict:
    scored = [g["score"] for g in grades if g["score"] >= 0]
    if not scored:
        return {}
    relevant = sum(1 for s in scored if s >= 1)
    return {
        "precision_at_k": relevant / top_k,
        "mean_relevance_score": sum(scored) / len(scored),
    }


def run_benchmark(
    run_dir: str,
    queries: list[dict],
    top_k: int = 5,
    llm_client: OpenRouterClient | None = None,
    llm_model: str = "openai/gpt-4o-mini",
    num_hops: int = 1,
    neighbor_weight: float = 0.5,
) -> list[dict]:
    """Run retrieval benchmark for all queries and return per-query result dicts."""
    graph, chunks, _ = load_graph_chunks_and_tree(run_dir)

    resolved = resolve_run_dir(run_dir)
    syn_table, can_kw, can_emb = load_canonicalization_data(resolved)
    canonical_lookup = (
        CanonicalLookup(syn_table, can_kw, can_emb) if syn_table is not None else None
    )

    retriever = KGNodeRetriever(
        graph,
        chunks,
        neighbor_weight=neighbor_weight,
        num_hops=num_hops,
        canonical_lookup=canonical_lookup,
    )

    results = []
    for q in queries:
        qid = q.get("id", "unknown")
        query_text = q.get("question", q.get("query", ""))
        ideal = set(q.get("ideal_retrieved_chunks", []))

        print(f"\n[{qid}] {query_text}")

        scores = retriever.get_scores(query_text, top_k, [])
        retrieved = sorted(
            [(cid, chunks[cid], score) for cid, score in scores.items() if cid in chunks],
            key=lambda x: x[2], reverse=True,
        )[:top_k]
        retrieved_ids = [cid for cid, _, _ in retrieved]

        if not retrieved:
            print("  WARNING: no chunks retrieved (no query nodes matched graph)")

        # Difficulty
        difficulty = None
        try:
            analysis = analyze_query(query_text, graph)
            difficulty = {
                "score": analysis.difficulty.score,
                "category": analysis.difficulty.category.value,
                "matched_nodes": analysis.features.query_node_count,
            }
            print(
                f"  Difficulty: {difficulty['category']} "
                f"(score={difficulty['score']}, nodes={difficulty['matched_nodes']})"
            )
        except Exception as e:
            logger.debug("Difficulty analysis failed for %r: %s", qid, e)

        # Ideal precision / recall
        ideal_m = None
        if ideal:
            ideal_m = _ideal_metrics(retrieved_ids, ideal, top_k)
            print(
                f"  Ideal  P@{top_k}={ideal_m['precision_at_k']:.2f}  "
                f"R@{top_k}={ideal_m['recall_at_k']:.2f}  "
                f"hits={ideal_m['hits']}"
            )

        # LLM grading
        llm_grades = None
        llm_m = None
        if llm_client and retrieved:
            try:
                llm_grades = _grade_with_llm(llm_client, llm_model, query_text, retrieved)
                llm_m = _llm_metrics(llm_grades, top_k)
                print(
                    f"  LLM    P@{top_k}={llm_m.get('precision_at_k', 0):.2f}  "
                    f"mean_score={llm_m.get('mean_relevance_score', 0):.2f}"
                )
            except Exception as e:
                logger.warning("LLM grading failed for %r: %s", qid, e)

        # Build annotated retrieved list
        retrieved_list = []
        for chunk_id, text, score in retrieved:
            entry: dict = {
                "chunk_id": chunk_id,
                "score": round(score, 4),
                "text_preview": text[:200],
            }
            if ideal:
                entry["in_ideal"] = chunk_id in ideal
            if llm_grades:
                grade = next((g for g in llm_grades if g["chunk_id"] == chunk_id), {})
                entry["llm_score"] = grade.get("score")
                entry["llm_reason"] = grade.get("reason", "")
            retrieved_list.append(entry)

        results.append(
            {
                "id": qid,
                "query": query_text,
                "difficulty": difficulty,
                "retrieved": retrieved_list,
                "ideal_metrics": ideal_m,
                "llm_metrics": llm_m,
            }
        )

    return results


def _avg(values: list[float]) -> float | None:
    clean = [v for v in values if v is not None]
    return sum(clean) / len(clean) if clean else None


def print_summary(results: list[dict], top_k: int) -> None:
    """Print an aggregate summary table to stdout."""
    has_ideal = any(r["ideal_metrics"] for r in results)
    has_llm = any(r["llm_metrics"] for r in results)

    col_id = 30
    cols = [("Query ID", col_id), ("Nodes", 5)]
    if has_ideal:
        cols += [(f"P@{top_k}", 6), (f"R@{top_k}", 6)]
    if has_llm:
        cols += [(f"LLM P@{top_k}", 8), ("LLM Mean", 8)]
    cols.append(("Difficulty", 10))

    header = "  ".join(f"{h:<{w}}" for h, w in cols)
    sep = "  ".join("-" * w for _, w in cols)

    print(f"\n{'=' * len(sep)}")
    print("BENCHMARK SUMMARY")
    print("=" * len(sep))
    print(header)
    print(sep)

    ideal_p, ideal_r, llm_p, llm_mean = [], [], [], []
    no_results = []

    for r in results:
        if not r["retrieved"]:
            no_results.append(r["id"])

        nodes = r["difficulty"]["matched_nodes"] if r["difficulty"] else "-"
        diff = r["difficulty"]["category"] if r["difficulty"] else "-"
        im = r["ideal_metrics"] or {}
        lm = r["llm_metrics"] or {}

        row = [(r["id"][:col_id], col_id), (str(nodes), 5)]
        if has_ideal:
            row += [
                (f"{im.get('precision_at_k', '-'):.2f}" if im else "-", 6),
                (f"{im.get('recall_at_k', '-'):.2f}" if im else "-", 6),
            ]
            if im:
                ideal_p.append(im["precision_at_k"])
                ideal_r.append(im["recall_at_k"])
        if has_llm:
            row += [
                (f"{lm.get('precision_at_k', '-'):.2f}" if lm else "-", 8),
                (f"{lm.get('mean_relevance_score', '-'):.2f}" if lm else "-", 8),
            ]
            if lm:
                llm_p.append(lm["precision_at_k"])
                llm_mean.append(lm["mean_relevance_score"])
        row.append((diff, 10))

        print("  ".join(f"{v:<{w}}" for v, w in row))

    print(sep)

    avg_row = [("AVERAGE", col_id), ("", 5)]
    if has_ideal:
        avg_p = _avg(ideal_p)
        avg_r = _avg(ideal_r)
        avg_row += [
            (f"{avg_p:.2f}" if avg_p is not None else "-", 6),
            (f"{avg_r:.2f}" if avg_r is not None else "-", 6),
        ]
    if has_llm:
        a_lp = _avg(llm_p)
        a_lm = _avg(llm_mean)
        avg_row += [
            (f"{a_lp:.2f}" if a_lp is not None else "-", 8),
            (f"{a_lm:.2f}" if a_lm is not None else "-", 8),
        ]
    avg_row.append(("", 10))
    print("  ".join(f"{v:<{w}}" for v, w in avg_row))

    if no_results:
        print(f"\nNo chunks retrieved (0 matched nodes): {', '.join(no_results)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark KGRetriever retrieval quality.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run-dir",
        default=RUNS_DIR,
        help="KG run directory or runs/ parent with 'latest' symlink",
    )
    parser.add_argument(
        "--queries",
        default="tests/benchmarks.yaml",
        help="YAML file with query list",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--model",
        default="openai/gpt-4o-mini",
        help="OpenRouter model for LLM grading",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="OpenRouter API key (falls back to OPENROUTER_API_KEY env var)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Write full results to this JSON file",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip LLM relevance grading",
    )
    parser.add_argument("--num-hops", type=int, default=1)
    parser.add_argument("--neighbor-weight", type=float, default=0.5)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    with open(args.queries) as f:
        data = yaml.safe_load(f)
    queries = data.get("benchmarks", data.get("queries", []))
    print(f"Loaded {len(queries)} queries from {args.queries}")

    llm_client = None
    if not args.no_llm:
        api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")
        if api_key:
            llm_client = OpenRouterClient(api_key, retries=2)
            print(f"LLM grading enabled: {args.model}")
        else:
            print(
                "No API key found — running without LLM grading. "
                "Pass --api-key or set OPENROUTER_API_KEY."
            )

    results = run_benchmark(
        run_dir=args.run_dir,
        queries=queries,
        top_k=args.top_k,
        llm_client=llm_client,
        llm_model=args.model,
        num_hops=args.num_hops,
        neighbor_weight=args.neighbor_weight,
    )

    print_summary(results, args.top_k)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nFull results written to {args.output}")


if __name__ == "__main__":
    load_dotenv()
    main()
