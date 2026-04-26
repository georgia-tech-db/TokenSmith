#!/usr/bin/env python3
"""
Inspect TokenSmith retrieval for a single query using the current local artifacts.
"""

from __future__ import annotations

import argparse
from typing import List

from src.config import RAGConfig, resolve_config_path
from src.retrieval_pipeline import build_runtime_retrievers, execute_retrieval_plan
from src.retriever import load_artifact_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect retrieval traces for a single query.")
    parser.add_argument("query", help="Question to run through retrieval.")
    parser.add_argument(
        "--history",
        nargs="*",
        default=[],
        help="Optional alternating user/assistant turns. Example: --history 'What is ARIES?' 'ARIES is a recovery algorithm.'",
    )
    parser.add_argument("--config", default="config/config.yaml", help="Path to config yaml.")
    parser.add_argument("--index-prefix", default="textbook_index", help="Artifact prefix.")
    parser.add_argument("--top-k", type=int, default=None, help="Override config top_k.")
    return parser.parse_args()


def _history_pairs(turns: List[str]) -> List[dict]:
    history = []
    roles = ["user", "assistant"]
    for index, turn in enumerate(turns):
        history.append({"role": roles[index % 2], "content": turn})
    return history


def main() -> None:
    args = parse_args()
    cfg = RAGConfig.from_yaml(resolve_config_path(args.config))
    if args.top_k is not None:
        cfg.top_k = args.top_k
    cfg.validate_runtime_files(require_index_sidecars=True)

    artifacts_dir = cfg.get_artifacts_directory()
    bundle = load_artifact_bundle(artifacts_dir, args.index_prefix)
    retrievers = build_runtime_retrievers(bundle, cfg)
    history = _history_pairs(args.history)

    ranked_chunks, chunk_ids, trace = execute_retrieval_plan(
        query=args.query,
        cfg=cfg,
        bundle=bundle,
        retrievers=retrievers,
        history=history,
    )

    print(f"Query: {args.query}")
    print(f"Query type: {trace.query_type} -> {trace.resolved_query_type}")
    print(f"Retrieval mode: {trace.retrieval_mode}")
    print(f"Route reason: {trace.route_reason}")
    if trace.rewritten_query:
        print(f"Rewritten query: {trace.rewritten_query}")
    if trace.sub_queries:
        print("Sub-queries:")
        for sub_query in trace.sub_queries:
            print(f"  - {sub_query}")
    if trace.selected_section_paths:
        print("Selected sections:")
        for section_path in trace.selected_section_paths:
            print(f"  - {section_path}")

    print("\nTop chunks:")
    for rank, chunk_id in enumerate(chunk_ids, start=1):
        meta = bundle.metadata[chunk_id]
        pages = meta.get("page_numbers", [])
        snippet = meta.get("raw_text", ranked_chunks[rank - 1]).replace("\n", " ")
        print(
            f"{rank:02d}. chunk={chunk_id} pages={pages} "
            f"section={meta.get('section_path', meta.get('section'))}"
        )
        print(f"    {snippet[:220]}")


if __name__ == "__main__":
    main()
