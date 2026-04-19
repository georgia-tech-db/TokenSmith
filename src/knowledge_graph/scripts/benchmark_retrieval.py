import argparse
import json
import logging
import os

import yaml
from dotenv import load_dotenv

from src.knowledge_graph.build import RUNS_DIR
from src.knowledge_graph.io import (
    load_canonicalization_data,
    load_graph_chunks_and_tree,
    resolve_run_dir,
)
from src.knowledge_graph.openrouter_client import OpenRouterClient
from src.knowledge_graph.query import (
    CanonicalLookup,
    KGNodeRetriever,
    SectionTreeRetriever,
)
from src.knowledge_graph.prompts import GRADE_PROMPT
from src.retriever import BM25Retriever, FAISSRetriever, IndexKeywordRetriever, load_artifacts

logger = logging.getLogger(__name__)


def _grade_with_llm(
    client: OpenRouterClient,
    model: str,
    query: str,
    retrieved: list[tuple[int, str, float]],
) -> list[dict]:
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


def _llm_metrics(grades: list[dict], top_k: int) -> dict:
    scored = [g["score"] for g in grades if g["score"] >= 0]
    if not scored:
        return {}
    relevant = sum(1 for s in scored if s >= 1)
    return {
        # Fraction of the top-k chunks judged relevant (score >= 1) by the LLM.
        "precision_at_k": relevant / top_k,
        # Average raw LLM relevance score across retrieved chunks (0=irrelevant, 1=partial, 2=relevant).
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
    artifacts_dir: str | None = None,
    index_prefix: str = "textbook_index",
    embed_model: str = "",
    extracted_index_path: str = "data/extracted_index.json",
    page_to_chunk_map_path: str = "index/sections/textbook_index_page_to_chunk_map.json",
) -> list[dict]:
    """Run retrieval benchmark for all queries across all available retrievers."""
    kg_graph, kg_chunks, tree = load_graph_chunks_and_tree(run_dir)

    resolved = resolve_run_dir(run_dir)
    syn_table, can_kw, can_emb = load_canonicalization_data(resolved)
    canonical_lookup = (
        CanonicalLookup(syn_table, can_kw, can_emb) if syn_table is not None else None
    )

    # Unified chunk lookup: RAG list takes precedence (dict-wrapped), KG dict as fallback.
    chunks: dict[int, str] = kg_chunks
    retrievers = []

    if artifacts_dir:
        try:
            faiss_idx, bm25_idx, rag_chunks, _, _ = load_artifacts(artifacts_dir, index_prefix)
            chunks = {i: t for i, t in enumerate(rag_chunks)}

            if embed_model:
                retrievers.append(FAISSRetriever(faiss_idx, embed_model))
                logger.info("FAISSRetriever enabled.")
            else:
                logger.info("Skipping FAISSRetriever: --embed-model not provided.")

            retrievers.append(BM25Retriever(bm25_idx))
            logger.info("BM25Retriever enabled.")

            if os.path.exists(extracted_index_path) and os.path.exists(page_to_chunk_map_path):
                retrievers.append(IndexKeywordRetriever(extracted_index_path, page_to_chunk_map_path))
                logger.info("IndexKeywordRetriever enabled.")
        except (FileNotFoundError, RuntimeError) as e:
            logger.warning("RAG artifacts not found, skipping FAISS/BM25: %s", e)

    retrievers.append(
        KGNodeRetriever(
            kg_graph,
            kg_chunks,
            neighbor_weight=neighbor_weight,
            num_hops=num_hops,
            canonical_lookup=canonical_lookup,
        )
    )

    if tree is not None:
        retrievers.append(SectionTreeRetriever(tree, kg_graph, canonical_lookup=canonical_lookup))
        logger.info("SectionTreeRetriever enabled.")
    else:
        logger.info("No section tree found — SectionTreeRetriever skipped.")

    results = []
    for q in queries:
        qid = q.get("id", "unknown")
        query_text = q.get("question", q.get("query", ""))

        print(f"\n[{qid}] {query_text}")

        retriever_results: dict[str, dict] = {}
        for retriever in retrievers:
            scores = retriever.get_scores(query_text, top_k, list(chunks.values()))
            retrieved = sorted(
                [(cid, chunks[cid], score) for cid, score in scores.items() if cid in chunks],
                key=lambda x: x[2],
                reverse=True,
            )[:top_k]
            if not retrieved:
                print(f"  [{retriever.name}] WARNING: no chunks retrieved")

            llm_grades = None
            llm_m = None
            if llm_client and retrieved:
                try:
                    llm_grades = _grade_with_llm(llm_client, llm_model, query_text, retrieved)
                    llm_m = _llm_metrics(llm_grades, top_k)
                    print(
                        f"  [{retriever.name}] LLM "
                        f"P@{top_k}={llm_m.get('precision_at_k', 0):.2f}  "
                        f"mean_score={llm_m.get('mean_relevance_score', 0):.2f}"
                    )
                except Exception as e:
                    logger.warning(
                        "LLM grading failed for %r / %r: %s", qid, retriever.name, e
                    )

            retrieved_list = []
            for chunk_id, text, score in retrieved:
                entry: dict = {
                    "chunk_id": chunk_id,
                    "score": round(score, 4),
                    "text_preview": text[:200],
                }
                if llm_grades:
                    grade = next((g for g in llm_grades if g["chunk_id"] == chunk_id), {})
                    entry["llm_score"] = grade.get("score")
                    entry["llm_reason"] = grade.get("reason", "")
                retrieved_list.append(entry)

            retriever_results[retriever.name] = {
                "retrieved": retrieved_list,
                "llm_metrics": llm_m,
            }

        results.append(
            {
                "id": qid,
                "query": query_text,
                "retrievers": retriever_results,
            }
        )

    return results


def _avg(values: list[float]) -> float | None:
    clean = [v for v in values if v is not None]
    return sum(clean) / len(clean) if clean else None


def print_summary(results: list[dict], top_k: int) -> None:
    retriever_names: list[str] = []
    for r in results:
        for name in r.get("retrievers", {}):
            if name not in retriever_names:
                retriever_names.append(name)

    col_id = 30
    col_w = 16  # width per retriever: "P@k  Mean" each 7 chars + spacing

    # Header row: Query ID + two sub-columns (P@k, Mean) per retriever
    header1 = f"{'Query ID':<{col_id}}"
    header2 = " " * col_id
    for name in retriever_names:
        short = name[:col_w].center(col_w)
        header1 += f"  {short}"
        sub = f"{'P@'+str(top_k):>6}  {'Mean':>6}"
        header2 += f"  {sub}"

    sep = "-" * len(header1)
    print(f"\n{sep}")
    print(header1)
    print(header2)
    print(sep)

    # Accumulators for averages
    agg: dict[str, dict[str, list[float]]] = {n: {"p": [], "m": []} for n in retriever_names}

    for r in results:
        row = f"{r['id'][:col_id]:<{col_id}}"
        for name in retriever_names:
            lm = r.get("retrievers", {}).get(name, {}).get("llm_metrics") or {}
            if lm:
                p = lm.get("precision_at_k", 0.0)
                m = lm.get("mean_relevance_score", 0.0)
                agg[name]["p"].append(p)
                agg[name]["m"].append(m)
                row += f"  {p:>6.2f}  {m:>6.2f}"
            else:
                row += f"  {'—':>6}  {'—':>6}"
        print(row)

    print(sep)
    avg_row = f"{'AVERAGE':<{col_id}}"
    for name in retriever_names:
        a_p = _avg(agg[name]["p"])
        a_m = _avg(agg[name]["m"])
        p_str = f"{a_p:.2f}" if a_p is not None else "—"
        m_str = f"{a_m:.2f}" if a_m is not None else "—"
        avg_row += f"  {p_str:>6}  {m_str:>6}"
    print(avg_row)
    print(sep)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark all retrievers against a query set.",
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
    parser.add_argument(
        "--artifacts-dir",
        default=None,
        help="RAG artifacts directory for FAISS/BM25 retrievers (e.g. index/recursive_sections/)",
    )
    parser.add_argument(
        "--index-prefix",
        default="textbook_index",
        help="Index artifact prefix used when building the RAG index",
    )
    parser.add_argument(
        "--embed-model",
        default="",
        help="Embedding model path for FAISSRetriever (GGUF or HuggingFace name)",
    )
    parser.add_argument(
        "--extracted-index",
        default="data/extracted_index.json",
        help="Path to extracted_index.json for IndexKeywordRetriever",
    )
    parser.add_argument(
        "--page-chunk-map",
        default="index/sections/textbook_index_page_to_chunk_map.json",
        help="Path to page_to_chunk_map.json for IndexKeywordRetriever",
    )
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
        artifacts_dir=args.artifacts_dir,
        index_prefix=args.index_prefix,
        embed_model=args.embed_model,
        extracted_index_path=args.extracted_index,
        page_to_chunk_map_path=args.page_chunk_map,
    )

    print_summary(results, args.top_k)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=lambda o: int(o) if hasattr(o, "__index__") else str(o))
        print(f"\nFull results written to {args.output}")


if __name__ == "__main__":
    load_dotenv()
    main()
