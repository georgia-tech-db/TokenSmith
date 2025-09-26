import argparse

from src.config import QueryPlanConfig
from src.instrumentation.logging import init_logger, get_logger
from src.planning.heuristics import HeuristicQueryPlanner
from src.preprocess import build_index
from src.ranking.ensemble import EnsembleRanker
from src.ranking.rankers import FaissSimilarityRanker, BM25Ranker, TfIDFRanker
from src.reranker import rerank
from src.retriever import get_candidates, apply_seg_filter
from src.generator  import answer

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["index", "chat"])
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--pdf_dir", default="data/chapters/")
    p.add_argument("--model_path", default="models/qwen2.5-0.5b-instruct-q5_k_m.gguf")

    # Extra indexing knobs
    p.add_argument("--pdf_range", type=str, default=None, help="e.g., 27-33")
    p.add_argument("--keep_tables", action="store_true")
    p.add_argument("--visualize", action="store_true")

    return p.parse_args()

def main():
    args = parse_args()
    cfg = QueryPlanConfig.from_yaml(args.config)
    init_logger(cfg)
    logger = get_logger()
    planner = HeuristicQueryPlanner(cfg)

    if args.mode == "index":
        # Optional range filtering
        if args.pdf_range:
            start, end = map(int, args.pdf_range.split("-"))
            pdf_paths = [f"{i}.pdf" for i in range(start, end)]
        else:
            pdf_paths = None

        build_index(
            pdf_dir=args.pdf_dir,
            out_prefix=cfg.index_prefix,
            cfg=cfg,
            keep_tables=args.keep_tables,
            pdf_files=pdf_paths,
            do_visualize=args.visualize
        )
        print("Index built ✓")

    elif args.mode == "chat":
        from src.retriever import load_artifacts

        print("📚 Ready. Type 'exit' to quit.")
        while True:
            q = input("\nAsk > ").strip()
            if q.lower() in {"exit","quit"}:
                break
            logger.log_query_start(q)
            cfg = planner.plan(q)
            index, chunks, sources, vectorizer, chunk_tags = \
                load_artifacts(cfg.index_prefix, cfg)

            pool_n = max(cfg.pool_size, cfg.top_k + 10)
            cand_idxs, faiss_dists = get_candidates(
                q, pool_n, index, chunks,
                embed_model=cfg.embed_model,
            )
            logger.log_retrieval(cand_idxs, faiss_dists, pool_n, cfg.embed_model)

            # 2) shared context for various rankers
            context = {
                "faiss_distances": faiss_dists, # for FaissSimilarityRanker
                "vectorizer": vectorizer,  # for TfIDFRanker
                "chunk_tags": chunk_tags,  # for TfIDFRanker
            }

            # 3) build rankers + ensemble (using weights from config)
            rankers = [
                FaissSimilarityRanker(),
                BM25Ranker(),
                TfIDFRanker(),
            ]
            weights = cfg.ranker_weights
            method = cfg.ensemble_method
            rrf_k = int(cfg.rrf_k)

            ensemble = EnsembleRanker(method, rankers, weights, rrf_k=rrf_k)
            ordered = ensemble.rank(query=q, chunks=chunks, cand_idxs=cand_idxs, context=context)

            topk_idxs = apply_seg_filter(cfg, chunks, ordered)
            logger.log_chunks_used(topk_idxs, chunks, sources, chunk_tags)

            # 4) materialize indices into text and continue
            ranked_chunks = [chunks[i] for i in topk_idxs]

            # 5) final re-ranking step before generation.
            reranked_chunks = rerank(q, ranked_chunks, mode=cfg.get("rerank_mode", ""), top_n=cfg.get("rerank_top_n", 20))

            # 6) query transformations could be applied here. See src/query_transform.py.

            ans = answer(
                q, reranked_chunks, args.model_path,
                max_tokens=cfg.max_gen_tokens,
            )
            print("\n=== ANSWER =========================================\n")
            print(ans if ans.strip() else "(no output)")
            print("\n====================================================\n")
            logger.log_generation(
                ans,
                {"max_tokens": cfg.max_gen_tokens, "model_path": args.model_path}
            )

        logger.log_query_complete()

if __name__ == "__main__":
    main()