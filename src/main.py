import argparse, yaml, pathlib
from typing import Optional

from src.config import QueryPlanConfig
from src.instrumentation.logging import init_logger, get_logger
from src.planning.heuristics import HeuristicQueryPlanner
from src.preprocess import build_index
from src.ranking.ensemble import EnsembleRanker
from src.ranking.rankers import FaissSimilarityRanker, BM25Ranker, TfIDFRanker
from src.retriever import get_candidates, apply_seg_filter
from src.ranker import rerank
from src.generator  import answer
from src.feedback_db import FeedbackDB, FeedbackEntry

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["index", "chat"])
    p.add_argument(
        "--config", default=None, required=False
    )  # default unnecessary as fallback is loaded
    p.add_argument("--pdf_dir", default="data/chapters/")
    p.add_argument("--index_prefix", default="textbook_index")
    p.add_argument(
        "--model_path", default=None, required=False
    )  # default set by config

    # Extra indexing knobs
    p.add_argument("--pdf_range", type=str, default=None, help="e.g., 27-33")
    p.add_argument("--keep_tables", action="store_true")
    p.add_argument("--visualize", action="store_true")

    return p.parse_args()


def load_correct_fallback_config_file() -> Optional[any]:
    user_config = pathlib.Path("~/.config/tokensmith/config.yaml")
    user_config_alt = pathlib.Path("~/.config/tokensmith/config.yml")
    default_config = pathlib.Path("config/config.yaml")

    if user_config.exists():
        return QueryPlanConfig.from_yaml(user_config)

    if user_config_alt.exists():
        return QueryPlanConfig.from_yaml(user_config_alt)

    if default_config.exists():
        return QueryPlanConfig.from_yaml(default_config)

    return None


def main():
    args = parse_args()

    # load config file from argument. If none provided, open fallback
    cfg = None
    if args.config is not None:
        cfg = QueryPlanConfig.from_yaml(args.config)
    else:
        cfg = load_correct_fallback_config_file()

    if cfg is None:
        raise ValueError(
            "Default config file not found. Expected at config/config.yaml"
        )

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
            do_visualize=args.visualize,
        )
        print("Index built ✓")

    elif args.mode == "chat":
        from src.retriever import load_artifacts
        db = FeedbackDB()

        print("📚 Ready. Type 'exit' to quit.")
        while True:
            q = input("\nAsk > ").strip()
            if q.lower() in {"exit", "quit"}:
                break
            logger.log_query_start(q)
            cfg = planner.plan(q)
            index, chunks, sources, vectorizer, chunk_tags = load_artifacts(
                cfg.index_prefix, cfg
            )

            pool_n = max(cfg.pool_size, cfg.top_k + 10)
            cand_idxs, faiss_dists = get_candidates(
                q,
                pool_n,
                index,
                chunks,
                embed_model=cfg.embed_model,
            )
            logger.log_retrieval(cand_idxs, faiss_dists, pool_n, cfg.embed_model)

            # 2) shared context for various rankers
            context = {
                "faiss_distances": faiss_dists,  # for FaissSimilarityRanker
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
            ordered = ensemble.rank(
                query=q, chunks=chunks, cand_idxs=cand_idxs, context=context
            )

            topk_idxs = apply_seg_filter(cfg, chunks, ordered)
            logger.log_chunks_used(topk_idxs, chunks, sources, chunk_tags)

            # 4) materialize indices into text and continue
            ranked_chunks = [chunks[i] for i in topk_idxs]

            # HALO Stub (NO OP for now)
            ranked_chunks = rerank(q, ranked_chunks, mode=cfg.halo_mode)

            def _collect_and_save_feedback(answer_text: str, style: str):
                print("Provide feedback: [u] thumbs up, [d] thumbs down, [enter] skip")
                fb_thumb_local = input("Thumbs (u/d or enter): ").strip().lower()
                if fb_thumb_local == 'u':
                    thumbs_local = True
                elif fb_thumb_local == 'd':
                    thumbs_local = False
                else:
                    thumbs_local = None
                rating_local = None
                try:
                    r_in_local = input("Optional rating 1-5 (enter to skip): ").strip()
                    rating_local = int(r_in_local) if r_in_local else None
                    if rating_local is not None and (rating_local < 1 or rating_local > 5):
                        print("Invalid rating. Skipping rating.")
                        rating_local = None
                except ValueError:
                    print("Invalid rating. Skipping rating.")
                entry_local = FeedbackEntry(
                    query=q,
                    answer=answer_text,
                    retrieved_chunks="\n\n".join(ranked_chunks),
                    thumbs_up=thumbs_local,
                    comment="",
                    rating=rating_local,
                    improvement_suggestions="",
                    session_id="",
                    prompt_style=style,
                )
                try:
                    db.add_feedback(entry_local)
                except Exception as e:
                    print(f"Warning: failed to save feedback: {e}")

            # Generate initial answer
            current_style = "default"
            ans = answer(
                q,
                ranked_chunks,
                args.model_path or cfg.model_path,
                max_tokens=cfg.max_gen_tokens,
                style=current_style,
            )
            print("\n=== ANSWER =========================================\n")
            print(ans if ans.strip() else "(no output)")
            print("\n====================================================\n")
            logger.log_generation(
                ans, {"max_tokens": cfg.max_gen_tokens, "model_path": args.model_path}
            )
            _collect_and_save_feedback(ans, current_style)

            # Regeneration loop
            while True:
                regen = input("Refine? [c] concise, [v] verbose, [n] no/skip: ").strip().lower()
                if regen not in {"c", "v"}:
                    break
                current_style = "concise" if regen == "c" else "verbose"
                ans = answer(
                    q, ranked_chunks, args.model_path or cfg.model_path,
                    max_tokens=cfg.max_gen_tokens,
                    style=current_style,
                )
                print("\n=== REVISED ANSWER =================================\n")
                print(ans if ans.strip() else "(no output)")
                print("\n====================================================\n")
                logger.log_generation(
                    ans,
                    {"max_tokens": cfg.max_gen_tokens, "model_path": args.model_path, "style": current_style}
                )
                _collect_and_save_feedback(ans, current_style)

        logger.log_query_complete()


if __name__ == "__main__":
    main()
