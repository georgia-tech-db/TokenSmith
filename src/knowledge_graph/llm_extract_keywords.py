import os
import json
import argparse
import pickle
import logging
from typing import List

from src.knowledge_graph.models import Chunk
from src.knowledge_graph.extractors.openrouter_extractor import OpenRouterExtractor

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_all_chunks(
    chunks_path: str,
    meta_path: str,
    chapter_filter: str = None,
    exclude_chapters: List[str] = None,
    chunk_ids: List[int] = None,
) -> List[Chunk]:
    """Load pre-chunked text and metadata from pickle files into Chunk objects.

    If chunk_ids is provided, only those specific chunks are returned.
    If chapter_filter is provided, only chunks from that chapter (based on section_path)
    are included.
    If exclude_chapters is provided, chunks from those chapters are skipped.
    """
    if not os.path.exists(chunks_path) or not os.path.exists(meta_path):
        logger.error(f"Files not found: {chunks_path} or {meta_path}")
        return []

    with open(chunks_path, "rb") as f:
        texts: list[str] = pickle.load(f)

    with open(meta_path, "rb") as f:
        metas: list[dict] = pickle.load(f)

    if len(texts) != len(metas):
        raise ValueError(
            f"Mismatch: {len(texts)} chunks vs {len(metas)} metadata entries"
        )

    chunks = []
    for i, (text, meta) in enumerate(zip(texts, metas)):
        chunk_id = meta.get("chunk_id", i)
        section_path = meta.get("section_path")

        if section_path is None:
            raise ValueError(f"Missing section_path in metadata for chunk {chunk_id}")

        # Check exclusion first
        if exclude_chapters:
            if any(section_path.startswith(ex) for ex in exclude_chapters):
                continue

        if chunk_ids is not None and chunk_id not in chunk_ids:
            continue

        if chapter_filter:
            if section_path.startswith(chapter_filter):
                chunks.append(Chunk(id=chunk_id, text=text, metadata=meta))
        else:
            chunks.append(Chunk(id=chunk_id, text=text, metadata=meta))
    return chunks


def main():
    # Use paths from run_kg_pipeline if possible, otherwise hardcode defaults or use args
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    default_chunks_pkl = os.path.join(
        project_root, "index", "sections", "textbook_index_chunks.pkl"
    )
    default_meta_pkl = os.path.join(
        project_root, "index", "sections", "textbook_index_meta.pkl"
    )
    output_path = os.path.join(
        project_root, "data", "knowledge_graph", "{chapter}__{model}__extractions.json"
    )

    parser = argparse.ArgumentParser(
        description="Extract keywords for all chunks using OpenRouter."
    )
    parser.add_argument("--api_key", required=True, help="OpenRouter API key")
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
        help="Number of keywords to extract per chunk",
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
        help="Specific chunk IDs to extract. If provided, other filters still apply but only these IDs will be considered.",
    )

    args = parser.parse_args()

    chapter_str = f"{args.chapter}" if args.chapter else "all"
    if args.exclude_chapters:
        chapter_str += f"_exc_{'_'.join(map(str, args.exclude_chapters))}"
    if args.chunk_ids:
        ids_label = "_".join(map(str, args.chunk_ids))
        if len(ids_label) > 30:
            ids_label = f"{len(args.chunk_ids)}ids"
        chapter_str += f"_ids_{ids_label}"

    output_path = output_path.format(
        chapter=chapter_str, model=args.model.replace("/", "_")
    )

    logger.info(f"Loading chunks from {args.chunks_path}...")
    try:
        chapter_filter = f"Chapter {args.chapter} " if args.chapter else None
        exclude_filters = [f"Chapter {c} " for c in args.exclude_chapters]
        chunks = load_all_chunks(
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
    extractor = OpenRouterExtractor(api_key=args.api_key, model=args.model)

    # Process extraction
    results = extractor.extract(chunks)

    # Format for JSON storage
    output_data = []
    for res in results:
        # Finding the original chunk to include some context if needed, but the request asked to store output.
        # Minimal storage: chunk_id and keywords.
        output_data.append({"chunk_id": res.chunk_id, "keywords": res.nodes})

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"Extraction complete. Results saved to {output_path}")


if __name__ == "__main__":
    main()
