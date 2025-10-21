import argparse
import pathlib
import sys
from typing import Dict, Optional

from src.config import QueryPlanConfig
from src.generator import answer
from src.index_builder import build_index
from src.instrumentation.logging import init_logger, get_logger
from src.ranking.ranker import EnsembleRanker
from src.preprocessing.chunking import DocumentChunker
from src.retriever import apply_seg_filter, BM25Retriever, FAISSRetriever, load_artifacts, load_summary_artifacts, SummaryRetriever
from src.query_enhancement import generate_hypothetical_document


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the application."""
    parser = argparse.ArgumentParser(
        description="Welcome to TokenSmith!"
    )

    # Required arguments
    parser.add_argument(
        "mode",
        choices=["index", "chat"],
        help="operation mode: 'index' to build index, 'chat' to query"
    )

    # Common arguments
    parser.add_argument(
        "--config",
        help="path to custom config file (uses default if not specified)"
    )
    parser.add_argument(
        "--pdf_dir",
        default="data/chapters/",
        help="directory containing PDF files (default: %(default)s)"
    )
    parser.add_argument(
        "--index_prefix",
        default="textbook_index",
        help="prefix for generated index files (default: %(default)s)"
    )
    parser.add_argument(
        "--model_path",
        help="path to generation model (uses config default if not specified)"
    )

    # Indexing-specific arguments
    indexing_group = parser.add_argument_group("indexing options")
    indexing_group.add_argument(
        "--pdf_range",
        metavar="START-END",
        help="specific range of PDFs to index (e.g., '27-33')"
    )
    indexing_group.add_argument(
        "--keep_tables",
        action="store_true",
        help="include tables in the index"
    )
    indexing_group.add_argument(
        "--visualize",
        action="store_true",
        help="generate visualizations during indexing"
    )

    return parser.parse_args()


def find_fallback_config() -> Optional[QueryPlanConfig]:
    """
    Looks for a fallback config file in standard locations.

    Returns:
        A QueryPlanConfig object if a file is found, otherwise None.
    """
    search_paths = [
        pathlib.Path("~/.config/tokensmith/config.yaml").expanduser(),
        pathlib.Path("~/.config/tokensmith/config.yml").expanduser(),
        pathlib.Path("config/config.yaml"),
    ]
    for path in search_paths:
        if path.exists():
            return QueryPlanConfig.from_yaml(path)
    return None


def run_index_mode(args: argparse.Namespace, cfg: QueryPlanConfig):
    """Handles the logic for building the index."""

    # Robust range filtering
    try:
        if args.pdf_range:
            start, end = map(int, args.pdf_range.split("-"))
            pdf_paths = [f"{i}.pdf" for i in range(start, end + 1)] # Inclusive range
            print(f"Indexing PDFs in range: {start}-{end}")
        else:
            pdf_paths = None
    except ValueError:
        print(f"ERROR: Invalid format for --pdf_range. Expected 'start-end', but got '{args.pdf_range}'.")
        sys.exit(1)
    
    strategy = cfg.make_strategy()
    chunker = DocumentChunker(strategy=strategy, keep_tables=args.keep_tables)
    
    artifacts_dir = cfg.make_artifacts_directory()

    build_index(
        markdown_file="data/book_without_image.md",
        chunker=chunker,
        chunk_config=cfg.chunk_config,
        embedding_model_path=cfg.embed_model,
        artifacts_dir=artifacts_dir,
        index_prefix=args.index_prefix,
        do_visualize=args.visualize,
        generate_summaries=cfg.generate_summaries,
        summary_model_path=cfg.model_path,
    )


def run_chat_session(args: argparse.Namespace, cfg: QueryPlanConfig):
    """
    Initializes artifacts and runs the main interactive chat loop.
    """
    logger = get_logger()

    print("Welcome to Tokensmith! Initializing chat...")
    try:
        artifacts_dir = cfg.make_artifacts_directory()
        faiss_index, bm25_index, chunks, sources = load_artifacts(artifacts_dir=artifacts_dir, index_prefix=args.index_prefix)

        retrievers = [
            FAISSRetriever(faiss_index, cfg.embed_model),
            BM25Retriever(bm25_index)
        ]
        ranker = EnsembleRanker(
            ensemble_method =cfg.ensemble_method,
            weights=cfg.ranker_weights,
            rrf_k=int(cfg.rrf_k)
        )

        summary_retriever = None
        if cfg.use_summaries:
            summary_index, summaries = load_summary_artifacts(artifacts_dir, args.index_prefix)
            if summary_index and summaries:
                summary_retriever = SummaryRetriever(summary_index, summaries, cfg.embed_model)
                print(f"Loaded {len(summaries)} section summaries")
            else:
                print("Summary retrieval enabled but no summaries found. Run indexing with generate_summaries=True")
                
    except Exception as e:
        print(f"ERROR: Failed to initialize chat artifacts: {e}")
        print("Please ensure you have run 'index' mode first.")
        sys.exit(1)

    print("Initialization complete. You can start asking questions!")
    print("Type 'exit' or 'quit' to end the session.")
    while True:
        try:
            q = input("\nAsk > ").strip()
            if not q:
                continue
            if q.lower() in {"exit", "quit"}:
                print("Goodbye!")
                break

            logger.log_query_start(q)

            retrieval_query = q
            if cfg.use_hyde:
                model_path = args.model_path or cfg.model_path
                hypothetical_doc = generate_hypothetical_document(
                    q, model_path, max_tokens=cfg.hyde_max_tokens
                )
                retrieval_query = hypothetical_doc
                print(f"HyDE query: {hypothetical_doc}")

            pool_n = max(cfg.pool_size, cfg.top_k + 10)
            raw_scores: Dict[str, Dict[int, float]] = {}
            for retriever in retrievers:
                raw_scores[retriever.name] = retriever.get_scores(retrieval_query, pool_n, chunks)
            logger.log_retrieval(
                candidates=sorted({idx for scores in raw_scores.values() for idx in scores}),
                retriever_scores={name: dict(scores) for name, scores in raw_scores.items()},
                pool_size=pool_n,
                embed_model=cfg.embed_model,
            )

            ordered = ranker.rank(raw_scores=raw_scores)
            topk_idxs = apply_seg_filter(cfg, chunks, ordered)
            logger.log_chunks_used(topk_idxs, chunks, sources)
            ranked_chunks = [chunks[i] for i in topk_idxs]

            top_summaries = None
            if summary_retriever:
                top_summaries = summary_retriever.get_top_summaries(retrieval_query, top_k=cfg.num_summaries)
                if top_summaries:
                    print(f"Using {len(top_summaries)} summaries + {len(ranked_chunks)} chunks")

            model_path = args.model_path or cfg.model_path
            ans = answer(q, ranked_chunks, model_path, max_tokens=cfg.max_gen_tokens, summaries=top_summaries)

            print("\n=================== START OF ANSWER ===================")
            print(ans.strip() if ans and ans.strip() else "(No output from model)")
            print("\n==================== END OF ANSWER ====================")
            logger.log_generation(ans, {"max_tokens": cfg.max_gen_tokens, "model_path": model_path})
            logger.log_query_complete()

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
            logger.log_error(e)
            break



def main():
    """Main entry point for the script."""
    args = parse_args()

    # Robust config loading
    if args.config:
        cfg = QueryPlanConfig.from_yaml(args.config)
    else:
        cfg = find_fallback_config()

    if cfg is None:
        raise FileNotFoundError(
            "No config file provided and no fallback found at config/ or ~/.config/tokensmith/"
        )

    init_logger(cfg)

    if args.mode == "index":
        run_index_mode(args, cfg)
    elif args.mode == "chat":
        run_chat_session(args, cfg)


if __name__ == "__main__":
    main()
