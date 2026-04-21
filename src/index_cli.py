"""CLI entry for index mode (separate from main to avoid heavy imports in lightweight tests)."""

import argparse
import pathlib
import sys
from typing import Callable, Optional

from src.config import RAGConfig
from src.preprocessing.chunking import DocumentChunker


def run_index_mode(
    args: argparse.Namespace,
    cfg: RAGConfig,
    *,
    build_index_fn: Optional[Callable[..., None]] = None,
) -> None:
    if build_index_fn is None:
        from src.index_builder import build_index as build_index_fn

    strategy = cfg.get_chunk_strategy()
    chunker = DocumentChunker(strategy=strategy, keep_tables=args.keep_tables)
    artifacts_dir = cfg.get_artifacts_directory()

    data_dir = pathlib.Path("data")
    print(f"Looking for markdown files in {data_dir.resolve()}...")
    md_files = sorted(data_dir.glob("*.md"))
    print(f"Found {len(md_files)} markdown files.")
    print(f"First 5 markdown files: {[str(f) for f in md_files[:5]]}")

    if not md_files:
        print("ERROR: No markdown files found in data/.", file=sys.stderr)
        sys.exit(1)

    build_index_fn(
        markdown_file=str(md_files[0]),
        chunker=chunker,
        chunk_config=cfg.chunk_config,
        embedding_model_path=cfg.embed_model,
        artifacts_dir=artifacts_dir,
        index_prefix=args.index_prefix,
        catalog_db_path=cfg.catalog_db_path,
        use_multiprocessing=args.multiproc_indexing,
        use_headings=args.embed_with_headings,
    )
