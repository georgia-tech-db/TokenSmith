"""
Vector search + optional Segment/Filter rules.

- Searches FAISS for top-k over ALL chunks
- If seg_filter is provided, filter the ranked results post-hoc (preserves FAISS ranking)
- Prints the first 100 characters of each returned chunk for debugging
"""

from __future__ import annotations
import pickle, faiss, numpy as np
from typing import Callable, List, Tuple, Optional
from sentence_transformers import SentenceTransformer
import os
from tagging import query_top_tags, tag_affinity_score

# cache the embedding model
_EMBED_CACHE: dict[str, SentenceTransformer] = {}


def _get_embedder(model_name: str) -> SentenceTransformer:
    if model_name not in _EMBED_CACHE:
        _EMBED_CACHE[model_name] = SentenceTransformer(model_name, device="cpu")
    return _EMBED_CACHE[model_name]

def load_artifacts(index_prefix: str) -> Tuple[faiss.Index, List[str], List[str], object, Optional[List[List[str]]]]:
    index   = faiss.read_index(f"{index_prefix}.faiss")
    chunks  = pickle.load(open(f"{index_prefix}_chunks.pkl", "rb"))
    sources = pickle.load(open(f"{index_prefix}_sources.pkl", "rb"))
    # Tagging artifacts (optional)
    try:
        vectorizer = pickle.load(open(os.path.join("meta", f"{index_prefix}_tfidf.pkl"), "rb"))
        chunk_tags = pickle.load(open(os.path.join("meta", f"{index_prefix}_tags.pkl"), "rb"))
    except Exception:
        vectorizer, chunk_tags = None, None
    return index, chunks, sources, vectorizer, chunk_tags


def _print_preview(chunks: List[str], n_preview: int = 100) -> None:
    for i, c in enumerate(chunks, 1):
        snippet = (c or "")[:n_preview].replace("\n", " ")
        print(f"[retriever] top{i:02d} → {len(c)} chars | {snippet!r}")


def _print_preview_idxs(chunks: List[str], srcs: List[str], tags: Optional[List[List[str]]], idxs: List[int], n_preview: int = 100) -> None:
    for rank, i in enumerate(idxs, 1):
        snippet = (chunks[i] or "")[:n_preview].replace("\n", " ")
        show_tags = (tags[i][:5] if tags else [])
        print(f"[retriever] top{rank:02d} | src={srcs[i]} | tags={show_tags} | {len(chunks[i])} chars | {snippet!r}")

def _rrf(ranks: List[int], k: int = 60) -> float:
    return sum(1.0 / (k + r) for r in ranks)

def _tag_rerank(query: str, cand_idxs: List[int], vectorizer, chunk_tags) -> List[int]:
    if vectorizer is None or chunk_tags is None or not cand_idxs:
        return cand_idxs

    q_tags = query_top_tags(query, vectorizer, top_q=8)

    # score each candidate by tag overlap; keep FAISS order as tiebreak
    scored = []
    for r, i in enumerate(cand_idxs, 1):  # r = original FAISS rank (1-based)
        s = tag_affinity_score(chunk_tags[i], q_tags, mode="weighted", tag_weights=None)
        scored.append((i, s, r))

    # sort by tag score desc, then original FAISS rank asc
    scored.sort(key=lambda x: (-x[1], x[2]))

    faiss_rank = {i: r for (i, _, r) in scored}          # original FAISS ranks
    tag_order  = [i for (i, _, _) in scored]             # order after tag sort
    tag_rank   = {i: rank for rank, i in enumerate(tag_order, start=1)}

    # RRF fuse tag-order with FAISS order
    fused = [(i, _rrf([faiss_rank[i], tag_rank[i]])) for i in tag_order]
    fused.sort(key=lambda x: -x[1])
    return [i for (i, _) in fused]

def retrieve(
    query: str,
    k: int,
    index: faiss.Index,
    chunks: List[str],
    *,
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    seg_filter: Optional[Callable[[str], bool]] = None,
    preview: bool = True,
    sources: Optional[List[str]] = None,
    vectorizer=None,
    chunk_tags: Optional[List[List[str]]] = None,
) -> List[str]:
    """
    Return up to k chunks for the query.

    Strategy:
      1) Search FAISS over the full corpus to get a ranked candidate list.
      2) If seg_filter is provided, keep only candidates where seg_filter(chunk) == True.
         If fewer than k survive, fall back to unfiltered to fill.
    """
    embedder = _get_embedder(embed_model)
    q_vec = embedder.encode([query]).astype("float32")
    overshoot = max(k * 5, k + 20)  # bigger pool so tags can help
    D, I = index.search(q_vec, overshoot)
    ranked_idxs = [i for i in I[0].tolist() if 0 <= i < len(chunks)]

    if seg_filter:
        ranked_idxs = [i for i in ranked_idxs if seg_filter(chunks[i])]
        if not ranked_idxs:
            return []

    # Tag-aware rerank
    ranked_idxs = _tag_rerank(query, ranked_idxs, vectorizer, chunk_tags)
    chosen = ranked_idxs[:k]

    if preview:
        if sources:
            _print_preview_idxs(chunks, sources, chunk_tags, chosen, n_preview=500)
        else:
            _print_preview([chunks[i] for i in chosen], n_preview=500)

    return [chunks[i] for i in chosen]
