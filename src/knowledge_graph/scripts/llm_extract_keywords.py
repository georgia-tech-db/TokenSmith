import os
import json
import argparse
import logging

from dotenv import load_dotenv

from src.knowledge_graph.build import (
    PROJECT_ROOT,
    EXTRACTIONS_DIR,
    LATEST_EXTRACTIONS,
    load_chunks,
)
from src.knowledge_graph.extractors.openrouter_extractor import OpenRouterExtractor

logger = logging.getLogger(__name__)


def _update_latest_symlink(target: str) -> None:
    """Point ``extractions/latest.json`` at *target* (absolute path)."""
    if os.path.islink(LATEST_EXTRACTIONS):
        os.unlink(LATEST_EXTRACTIONS)
    os.symlink(os.path.abspath(target), LATEST_EXTRACTIONS)


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    default_chunks_pkl = os.path.join(
        PROJECT_ROOT, "index", "sections", "textbook_index_chunks.pkl"
    )
    default_meta_pkl = os.path.join(
        PROJECT_ROOT, "index", "sections", "textbook_index_meta.pkl"
    )
    output_path_template = os.path.join(
        EXTRACTIONS_DIR, "{chapter}__{model}__extractions.json"
    )

    parser = argparse.ArgumentParser(
        description="Extract keywords for all chunks using OpenRouter."
    )
    parser.add_argument("--api_key", required=False, help="OpenRouter API key")
    parser.add_argument(
        "--chunks_path", default=default_chunks_pkl, help="Path to chunks.pkl"
    )
    parser.add_argument(
        "--meta_path", default=default_meta_pkl, help="Path to meta.pkl"
    )
    parser.add_argument(
        "--chapter",
        type=int,
        default=None,
        help="Chapter to process. If omitted, all chapters are processed (unless excluded).",
    )
    parser.add_argument(
        "--exclude_chapters",
        type=int,
        nargs="+",
        default=[],
        help="Chapters to exclude.",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=10,
        help="Number of keywords to extract per chunk (ignored when --adaptive_top_n is set)",
    )
    parser.add_argument(
        "--adaptive_top_n",
        action="store_true",
        default=False,
        help="Scale top_n per chunk as int(sqrt(len(chunk.text))) instead of a fixed value",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen/qwen3-next-80b-a3b-instruct",
        help="Model to use for extraction",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of chunks to process (for testing)",
    )
    parser.add_argument(
        "--chunk_ids",
        type=int,
        nargs="+",
        default=None,
        help="Specific chunk IDs to extract. If provided, other filters still apply but only these"
        " IDs will be considered.",
    )

    args = parser.parse_args()

    api_key = args.api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "OpenRouter API key not provided. Set it via --api_key or in the OPENROUTER_API_KEY"
            " environment variable."
        )

    chapter_str = f"{args.chapter}" if args.chapter else "all"
    if args.exclude_chapters:
        chapter_str += f"_exc_{'_'.join(map(str, args.exclude_chapters))}"
    if args.chunk_ids:
        ids_label = "_".join(map(str, args.chunk_ids))
        if len(ids_label) > 30:
            ids_label = f"{len(args.chunk_ids)}ids"
        chapter_str += f"_ids_{ids_label}"

    output_path = output_path_template.format(
        chapter=chapter_str, model=args.model.replace("/", "_")
    )

    logger.info(f"Loading chunks from {args.chunks_path}...")
    try:
        chapter_filter = f"Chapter {args.chapter} " if args.chapter else None
        exclude_filters = [f"Chapter {c} " for c in args.exclude_chapters]
        chunks = load_chunks(
            args.chunks_path,
            args.meta_path,
            chapter_filter=chapter_filter,
            exclude_chapters=exclude_filters,
            chunk_ids=args.chunk_ids,
        )
    except Exception as e:
        logger.error(f"Failed to load chunks: {e}")
        return

    if not chunks:
        logger.warning("No chunks loaded. Exiting.")
        return

    if args.limit:
        logger.info(f"Limiting to first {args.limit} chunks.")
        chunks = chunks[: args.limit]

    logger.info(
        f"Starting extraction for {len(chunks)} chunks using OpenRouterExtractor..."
    )
    extractor = OpenRouterExtractor(
        api_key=api_key,
        model=args.model,
        top_n=args.top_n,
        adaptive_top_n=args.adaptive_top_n,
    )

    results = extractor.extract(chunks)

    output_data = [
        {"chunk_id": res.chunk_id, "keywords": res.keywords} for res in results
    ]

    os.makedirs(EXTRACTIONS_DIR, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"Extraction complete. Results saved to {output_path}")

    _update_latest_symlink(output_path)
    logger.info(f"Updated: {LATEST_EXTRACTIONS} -> {output_path}")


if __name__ == "__main__":
    load_dotenv()
    main()
