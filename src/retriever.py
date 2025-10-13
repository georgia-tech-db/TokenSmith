"""
Vector search with optional BM25 re-ranking + Segment/Filter.

Pipeline:
  1) FAISS narrows to a candidate pool.
  2) (optional) BM25 re-ranks within that pool.
  3) (optional) seg_filter applied post-hoc (preserves ranking; backfills to k).
  4) Preview prints source.

Return: list[str] chunks (top-k).
"""

from __future__ import annotations
import pickle
from typing import List, Tuple, Optional, Dict

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.config import QueryPlanConfig


# -------------------------- Embedder cache ------------------------------

_EMBED_CACHE: Dict[str, SentenceTransformer] = {}

def _get_embedder(model_name: str) -> SentenceTransformer:
    if model_name not in _EMBED_CACHE:
        # Use the cached embedding model to avoid reloading it on every call
        _EMBED_CACHE[model_name] = SentenceTransformer(model_name, device="cpu")
    return _EMBED_CACHE[model_name]


# -------------------------- Artifacts I/O -------------------------------

def load_artifacts(cfg: QueryPlanConfig) -> Tuple[faiss.Index, List[str], List[str]]:
    """
    Loads:
      - FAISS index: {index_prefix}.faiss
      - chunks:      {index_prefix}_chunks.pkl
      - sources:     {index_prefix}_sources.pkl
    """
    index_prefix = cfg.get_index_prefix()

    faiss_index = faiss.read_index(f"{index_prefix}.faiss")
    bm25_index  = pickle.load(open(f"{index_prefix}_bm25.pkl", "rb"))
    chunks      = pickle.load(open(f"{index_prefix}_chunks.pkl", "rb"))
    sources     = pickle.load(open(f"{index_prefix}_sources.pkl", "rb"))

    return faiss_index, bm25_index, chunks, sources


# -------------------------- Pretty previews -----------------------------

def _print_preview(chunks: List[str], n_preview: int = 100) -> None:
    for i, c in enumerate(chunks, 1):
        snippet = (c or "")[:n_preview].replace("\n", " ")
        print(f"[retriever] top{i:02d} → {len(c)} chars | {snippet!r}")


def _print_preview_idxs(
    chunks: List[str],
    srcs: List[str],
    tags: Optional[List[List[str]]],
    idxs: List[int],
    n_preview: int = 100,
) -> None:
    for rank, i in enumerate(idxs, 1):
        snippet = (chunks[i] or "")[:n_preview].replace("\n", " ")
        show_tags = (tags[i][:5] if tags else [])
        print(f"[retriever] top{rank:02d} | src={srcs[i]} | tags={show_tags} | {len(chunks[i])} chars | {snippet!r}")


# -------------------------- Retrieval core ------------------------------

def get_faiss_candidates(
    query: str,
    pool_size: int,
    index: faiss.Index,
    chunks: List[str],
    *,
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> Dict[int, float]:
    """
    Returns faiss_distances for top 'pool_size' keyed by global chunk index.
    """
    embedder = _get_embedder(embed_model)

    # FAISS expects a 2D array
    q_vec = embedder.encode([query]).astype("float32")
    
    # Safety check on vector dimensions
    if q_vec.shape[1] != index.d:
        raise ValueError(
            f"Embedding dim mismatch: index={index.d} vs query={q_vec.shape[1]}"
        )

    # Perform the search
    distances, indices = index.search(q_vec, pool_size)

    # Remove invalid indices and ensure they are within bounds
    cand_idxs = [i for i in indices[0] if 0 <= i < len(chunks)]

    # Create the distance dictionary, ensuring we only include valid candidates
    dists = {idx: float(dist) for idx, dist in zip(cand_idxs, distances[0][:len(cand_idxs)])}

    return dists


def get_bm25_candidates(
    query: str,
    pool_size: int,
    index: BM25Okapi,
    chunks: List[str],
    *,
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> Dict[int, float]:
    """
    Returns bm25_scores for top 'pool_size' keyed by global chunk index.
    """
    # Tokenize the query in the same way the index was built
    tokenized_query = query.lower().split()

    # Get scores for all documents in the corpus
    all_scores = index.get_scores(tokenized_query)

    # Find the indices of the top 'pool_size' scores
    num_candidates = min(pool_size, len(all_scores))
    top_k_indices = np.argpartition(-all_scores, kth=num_candidates-1)[:num_candidates]
    
    # Get the corresponding scores for the top indices
    top_scores = all_scores[top_k_indices]

    # Format the output as a dictionary of scores
    scores = {int(idx): float(score) for idx, score in zip(top_k_indices, top_scores)}

    return scores


def apply_seg_filter(cfg: QueryPlanConfig, chunks, ordered):
    seg_filter = cfg.seg_filter
    if seg_filter:
        keep = [i for i in ordered if seg_filter(chunks[i])]
        back = [i for i in ordered if i not in keep]
        topk_idxs = (keep + back)[:cfg.top_k]
    else:
        topk_idxs = ordered[:cfg.top_k]
    return topk_idxs
