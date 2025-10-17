#!/usr/bin/env python3
"""Inspect FAISS and BM25 rankings for a single query. Example: python scripts/retrieval_diagnostics.py"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import pandas as pd

from pandas import DataFrame

# Add project root to sys.path for imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import QueryPlanConfig
from src.retriever import FAISSRetriever, BM25Retriever, load_artifacts


QUERY = "How does the recovery manager use ARIES to ensure atomicity?"
CONFIG_PATH: Optional[Path] = None
OUTPUT_PATH = Path("logs/retrieval_diagnostics.json")
INCLUDE_TEXT = True
POOL_SIZE_OVERRIDE: Optional[int] = None


def find_config(path: Optional[Path]) -> QueryPlanConfig:
    if path:
        return QueryPlanConfig.from_yaml(path)
    search = [
        Path("~/.config/tokensmith/config.yaml").expanduser(),
        Path("~/.config/tokensmith/config.yml").expanduser(),
        Path("config/config.yaml"),
    ]
    for candidate in search:
        if candidate.exists():
            return QueryPlanConfig.from_yaml(candidate)
    raise FileNotFoundError("Config not provided and no fallback discovered.")


def build_faiss_table(
    index,
    embed_model: str,
    query: str,
    chunks: List[str],
    sources: List[str],
    include_text: bool,
    pool_size: int,
) -> Tuple[List[Dict[str, object]], Optional[DataFrame]]:
    retriever = FAISSRetriever(index, embed_model)
    q_vec = retriever.embedder.encode([query]).astype("float32")
    distances, indices = index.search(q_vec, pool_size)
    records: List[Dict[str, object]] = []
    for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), start=1):
        if idx < 0 or idx >= len(chunks):
            continue
        item = {
            "chunk_idx": int(idx),
            "rank": rank,
            "distance": float(dist),
            "score": 1.0 / (1.0 + float(dist)),
            "source": sources[idx],
        }
        if include_text:
            item["text"] = chunks[idx]
        records.append(item)
    df = pd.DataFrame(records) if pd else None
    return records, df


def build_bm25_table(
    index,
    query: str,
    chunks: List[str],
    sources: List[str],
    include_text: bool,
    pool_size: int,
) -> Tuple[List[Dict[str, object]], Optional[DataFrame]]:
    retriever = BM25Retriever(index)
    score_map = retriever.get_scores(query, pool_size, chunks)
    ordered = sorted(score_map.items(), key=lambda kv: kv[1], reverse=True)

    records: List[Dict[str, object]] = []
    for rank, (idx, score) in enumerate(ordered[:pool_size], start=1):
        if idx < 0 or idx >= len(chunks):
            continue
        item = {
            "chunk_idx": int(idx),
            "rank": rank,
            "score": float(score),
            "source": sources[idx],
        }
        if include_text:
            item["text"] = chunks[idx]
        records.append(item)
    df = pd.DataFrame(records) if pd else None
    return records, df


def main() -> None:
    query = QUERY.strip()
    if not query:
        print("No query configured, exiting.")
        return

    cfg = find_config(CONFIG_PATH)

    faiss_index, bm25_index, chunks, sources = load_artifacts(cfg)
    pool_size = POOL_SIZE_OVERRIDE or len(chunks)

    faiss_records, faiss_df = build_faiss_table(
        faiss_index,
        cfg.embed_model,
        query,
        chunks,
        sources,
        INCLUDE_TEXT,
        pool_size,
    )
    bm25_records, bm25_df = build_bm25_table(
        bm25_index,
        query,
        chunks,
        sources,
        INCLUDE_TEXT,
        pool_size,
    )

    payload = {
        "query": query,
        "pool_size": pool_size,
        "faiss": faiss_records,
        "bm25": bm25_records,
    }
    if OUTPUT_PATH.suffix != ".json":
        raise ValueError("OUTPUT_PATH must end with .json")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2))

    print(f"Stored diagnostics at {OUTPUT_PATH}")
    if pd and faiss_records:
        print("FAISS head:\n", faiss_df.head())
    if pd and bm25_records:
        print("BM25 head:\n", bm25_df.head())


if __name__ == "__main__":
    main()
