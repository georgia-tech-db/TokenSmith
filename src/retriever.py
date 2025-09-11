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
import os
import pickle
from typing import Callable, List, Tuple, Optional, Dict

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from src.config import QueryPlanConfig
# Optional tag utilities (only used if vectorizer & chunk_tags are provided)
from src.ranking.tagging import query_top_tags, tag_affinity_score

# -------------------------- Embedder cache ------------------------------
_EMBED_CACHE: Dict[str, SentenceTransformer] = {}


def _get_embedder(model_name: str) -> SentenceTransformer:
    if model_name not in _EMBED_CACHE:
        _EMBED_CACHE[model_name] = SentenceTransformer(model_name, device="cpu")
    return _EMBED_CACHE[model_name]


# -------------------------- Artifacts I/O -------------------------------
def load_artifacts(index_prefix: str) -> Tuple[faiss.Index, List[str], List[str], object, Optional[List[List[str]]]]:
    """
    Loads:
      - FAISS index: {index_prefix}.faiss
      - chunks:      {index_prefix}_chunks.pkl
      - sources:     {index_prefix}_sources.pkl
      - meta/tagging (optional under meta/):
          meta/{index_prefix}_tfidf.pkl  -> vectorizer
          meta/{index_prefix}_tags.pkl   -> chunk_tags (List[List[str]])
    """
    index   = faiss.read_index(f"{index_prefix}.faiss")
    chunks  = pickle.load(open(f"{index_prefix}_chunks.pkl", "rb"))
    sources = pickle.load(open(f"{index_prefix}_sources.pkl", "rb"))

    try:
        vectorizer = pickle.load(open(os.path.join("meta", f"{index_prefix}_tfidf.pkl"), "rb"))
        chunk_tags = pickle.load(open(os.path.join("meta", f"{index_prefix}_tags.pkl"), "rb"))
    except Exception:
        vectorizer, chunk_tags = None, None

    return index, chunks, sources, vectorizer, chunk_tags


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
def retrieve(
    query: str,
    k: int,
    index: faiss.Index,
    chunks: List[str],
    *,
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    overshoot_factor: int = 3,
    fusion: str = "weighted",       # "weighted" | "rrf"
    rrf_k: int = 60,
    bm25_weight: float = 0.3,       # 0 = FAISS only, 1 = BM25 only (weighted mode)
    tag_weight: float = 0.2,        # 0 = disable tag rerank (weighted mode)
    seg_filter: Optional[Callable[[str], bool]] = None,
    preview: bool = True,
    sources: Optional[List[str]] = None,
    vectorizer=None,
    chunk_tags: Optional[List[List[str]]] = None,
    early_exit: bool =True
) -> List[str]:
    """
    Hybrid retriever:
      - FAISS → candidate pool (overshoot)
      - BM25 (optional) on candidate pool
      - Tag-aware rerank (optional) using TF-IDF tags
      - Fusion: weighted OR Reciprocal Rank Fusion (RRF)
      - seg_filter applied post-hoc; preserves ranking; backfills to k

    Returns: top-k chunk texts.
    """
    # --- Encode query & FAISS search ---
    embedder = _get_embedder(embed_model)
    q_vec = embedder.encode([query]).astype("float32")

    # Dim safety
    try:
        idx_dim = index.d
    except AttributeError:
        idx_dim = q_vec.shape[1]
    if q_vec.shape[1] != idx_dim:
        raise ValueError(
            f"Embedding dim mismatch: index={idx_dim} vs query={q_vec.shape[1]}. "
            f"Use the same embedding model used at index-build time."
        )

    overshoot = max(k * overshoot_factor, k + 5)
    D, I = index.search(q_vec, overshoot)

    cand_idxs = [i for i in I[0].tolist() if 0 <= i < len(chunks)]
    dists = {i: float(d) for i, d in zip(cand_idxs, D[0][:len(cand_idxs)])}
    if not cand_idxs:
        return []

    if early_exit:
        return cand_idxs, dists

    # Convert FAISS L2 distances → [0,1] similarity
    faiss_sims = 1.0 / (1.0 + D[0][: len(cand_idxs)])
    faiss_norm = (faiss_sims - np.min(faiss_sims)) / (np.ptp(faiss_sims) + 1e-8)
    faiss_score: Dict[int, float] = {idx: float(s) for idx, s in zip(cand_idxs, faiss_norm)}

    # --- BM25 on candidate pool (optional) ---
    bm25_score: Dict[int, float] = {idx: 0.0 for idx in cand_idxs}
    if fusion == "weighted" and bm25_weight > 0:
        tok_query = query.lower().split()
        cand_docs = [chunks[i].lower().split() for i in cand_idxs]
        bm25 = BM25Okapi(cand_docs)
        bm_sub = bm25.get_scores(tok_query)
        if np.max(bm_sub) > 0:
            bm_sub = bm_sub / (np.max(bm_sub) + 1e-8)
        else:
            bm_sub = np.zeros_like(bm_sub, dtype=float)
        bm25_score = {idx: float(s) for idx, s in zip(cand_idxs, bm_sub)}

    # --- Tag-aware score on candidate pool (optional) ---
    tag_score: Dict[int, float] = {idx: 0.0 for idx in cand_idxs}
    use_tags = (vectorizer is not None) and (chunk_tags is not None)
    if use_tags:
        q_tags = query_top_tags(query, vectorizer, top_q=8)
        raw = np.array(
            [tag_affinity_score(chunk_tags[i], q_tags, mode="weighted", tag_weights=None) for i in cand_idxs],
            dtype=float,
        )
        if np.max(raw) > 0:
            raw = (raw - np.min(raw)) / (np.ptp(raw) + 1e-8)
        else:
            raw = np.zeros_like(raw)
        tag_score = {idx: float(s) for idx, s in zip(cand_idxs, raw)}

    # --- Fusion ---
    if fusion == "weighted":
        w_bm  = max(0.0, min(1.0, bm25_weight))
        w_tag = max(0.0, min(1.0, tag_weight if use_tags else 0.0))
        w_faiss = max(0.0, 1.0 - w_bm - w_tag)

        combined = {
            idx: (w_faiss * faiss_score[idx]) + (w_bm * bm25_score.get(idx, 0.0)) + (w_tag * tag_score.get(idx, 0.0))
            for idx in cand_idxs
        }
        ranked_idxs = [i for i, _ in sorted(combined.items(), key=lambda x: x[1], reverse=True)]
    else:
        raise ValueError(f"Unknown fusion mode: {fusion!r}")

    # --- Post-hoc seg_filter that preserves ranking & backfills ---
    if seg_filter:
        keep = [i for i in ranked_idxs if seg_filter(chunks[i])]
        backfill = [i for i in ranked_idxs if i not in keep]
        final_idxs = (keep + backfill)[:k]
    else:
        final_idxs = ranked_idxs[:k]

    # --- Build result ---
    if preview:
        if sources:
            _print_preview_idxs(chunks, sources, chunk_tags, final_idxs, n_preview=500)
        else:
            _print_preview([chunks[i] for i in final_idxs], n_preview=500)

    return [chunks[i] for i in final_idxs]

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
