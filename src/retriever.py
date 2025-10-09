"""
Vector search with optional BM25 and Tag-aware re-ranking + Segment/Filter.

Pipeline:
  1) FAISS narrows to a candidate pool.
  2) (optional) BM25 re-ranks within that pool.
  3) (optional) Tag-aware re-rank using TF-IDF tags (query_top_tags, tag_affinity_score).
  4) (optional) seg_filter applied post-hoc (preserves ranking; backfills to k).
  5) Preview prints source + tags.

Return: list[str] chunks (top-k).
"""

from __future__ import annotations
import pickle
from typing import List, Tuple, Optional, Dict

import faiss
from src.embedder import Embedder

from src.config import QueryPlanConfig


# -------------------------- Embedder cache ------------------------------
_EMBED_CACHE: Dict[str, Embedder] = {}


def _get_embedder(model_name: str) -> Embedder:
    if model_name not in _EMBED_CACHE:
        _EMBED_CACHE[model_name] = Embedder(model_name, device="cpu")
    return _EMBED_CACHE[model_name]


# -------------------------- Artifacts I/O -------------------------------
def load_artifacts(index_prefix: str, cfg: QueryPlanConfig) -> Tuple[faiss.Index, List[str], List[str], object, Optional[List[List[str]]]]:
    """
    Loads:
      - FAISS index: {index_prefix}.faiss
      - chunks:      {index_prefix}_chunks.pkl
      - sources:     {index_prefix}_sources.pkl
      - meta/tagging (optional under meta/):
          meta/{index_prefix}_tfidf.pkl  -> vectorizer
          meta/{index_prefix}_tags.pkl   -> chunk_tags (List[List[str]])
    """
    faiss_prefix = cfg.get_faiss_prefix(index_prefix)
    meta_prefix = cfg.get_tfidf_prefix(index_prefix)

    index   = faiss.read_index(f"{faiss_prefix}.faiss")
    chunks  = pickle.load(open(f"{faiss_prefix}_chunks.pkl", "rb"))
    sources = pickle.load(open(f"{faiss_prefix}_sources.pkl", "rb"))

    try:
        vectorizer = pickle.load(open(f"{meta_prefix}_tfidf.pkl", "rb"))
        chunk_tags = pickle.load(open(f"{meta_prefix}_tags.pkl", "rb"))
    except Exception:
        vectorizer, chunk_tags = None, None

    return index, chunks, sources, vectorizer, chunk_tags


# -------------------------- Pretty previews -----------------------------
def print_preview(chunks: List[str]) -> None:
    n_preview = 3
    for i, c in enumerate(chunks, 1):
        snippet = (c or "")[:n_preview].replace("\n", " ")
        print(f"[retriever] top{i:02d} → {len(c)} chars | {snippet!r}")


def print_preview_idxs(
    chunks: List[str],
    srcs: List[str],
    tags: Optional[List[List[str]]],
    idxs: List[int],
) -> None:
    n_preview = 3
    for rank, i in enumerate(idxs, 1):
        snippet = (chunks[i] or "")[:n_preview].replace("\n", " ")
        show_tags = (tags[i][:5] if tags else [])
        print(f"[retriever] top{rank:02d} | src={srcs[i]} | tags={show_tags} | {len(chunks[i])} chars | {snippet!r}")

# -------------------------- Retrieval core ------------------------------
def get_candidates(
    query: str,
    pool_size: int,
    index: faiss.Index,
    chunks: List[str],
    *,
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> Tuple[List[int], Dict[int, float]]:
    """
    Returns (cand_idxs, faiss_distances) for top 'pool_size'.
    Distances keyed by global chunk index.
    """
    embedder = _get_embedder(embed_model)
    q_vec = embedder.encode([query]).astype("float32")

    # Safety on dims
    try:
        idx_dim = index.d
    except AttributeError:
        idx_dim = q_vec.shape[1]
    if q_vec.shape[1] != idx_dim:
        raise ValueError(f"Embedding dim mismatch: index={idx_dim} vs query={q_vec.shape[1]}")

    D, I = index.search(q_vec, pool_size)
    cand_idxs = [i for i in I[0].tolist() if 0 <= i < len(chunks)]
    dists = {i: float(d) for i, d in zip(cand_idxs, D[0][:len(cand_idxs)])}

    return cand_idxs, dists

def apply_seg_filter(cfg: QueryPlanConfig, chunks, ordered):
    seg_filter = cfg.seg_filter
    if seg_filter:
        keep = [i for i in ordered if seg_filter(chunks[i])]
        back = [i for i in ordered if i not in keep]
        topk_idxs = (keep + back)[:cfg.top_k]
    else:
        topk_idxs = ordered[:cfg.top_k]
    return topk_idxs
