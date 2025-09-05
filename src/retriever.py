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
from rank_bm25 import BM25Okapi

# cache the embedding model
_EMBED_CACHE: dict[str, SentenceTransformer] = {}


def _get_embedder(model_name: str) -> SentenceTransformer:
    if model_name not in _EMBED_CACHE:
        _EMBED_CACHE[model_name] = SentenceTransformer(model_name, device="cpu")
    return _EMBED_CACHE[model_name]


def load_artifacts(index_prefix: str) -> Tuple[faiss.Index, List[str]]:
    index  = faiss.read_index(f"{index_prefix}.faiss")
    chunks = pickle.load(open(f"{index_prefix}_chunks.pkl", "rb"))
    return index, chunks


def _print_preview(chunks: List[str], n_preview: int = 100) -> None:
    for i, c in enumerate(chunks, 1):
        snippet = (c or "")[:n_preview].replace("\n", " ")
        print(f"[retriever] top{i:02d} → {len(c)} chars | {snippet!r}")


def retrieve(
    query: str,
    k: int,
    index: faiss.Index,
    chunks: List[str],
    *,
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    bm25_weight: float = 0.3,
    overshoot_factor: int = 3,
    seg_filter: Optional[Callable[[str], bool]] = None,
    preview: bool = True,
) -> List[str]:
    """
    Hybrid retriever: FAISS narrows to candidate pool, BM25 re-ranks within that pool.

    Args:
        query: Student query string
        k: number of chunks to return
        index: FAISS index
        chunks: list of text chunks
        bm25: prebuilt BM25Okapi index (built once on all chunks)
        embed_model: embedding model for FAISS
        bm25_weight: interpolation weight (0=FAISS only, 1=BM25 only)
        overshoot_factor: how many extra FAISS candidates to fetch
        seg_filter: optional filter to drop irrelevant segments
        preview: whether to print retrieved previews
    """

    # --- FAISS search (candidate pool) ---
    embedder = _get_embedder(embed_model)
    q_vec = embedder.encode([query]).astype("float32")
    overshoot = max(k * overshoot_factor, k + 5)
    D, I = index.search(q_vec, overshoot)

    faiss_idxs = I[0].tolist()
    faiss_sims = [1 / (1 + d) for d in D[0]]  # L2 → similarity

    # Normalize FAISS similarities to [0,1]
    faiss_norm = (np.array(faiss_sims) - np.min(faiss_sims)) / (np.ptp(faiss_sims) + 1e-8)
    faiss_dict = {idx: score for idx, score in zip(faiss_idxs, faiss_norm)}

    # --- BM25 search (only over candidate pool) ---
    tokenized_query = query.lower().split()
    tokenized_chunks = [c.lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    bm25_scores = bm25.get_scores(tokenized_query)

    # Normalize BM25 scores to [0,1]
    if np.max(bm25_scores) > 0:
        bm25_norm = bm25_scores / (np.max(bm25_scores) + 1e-8)
    else:
        bm25_norm = np.zeros(len(chunks))

    # --- Combine scores (only candidate pool) ---
    combined_scores = {}
    for idx in faiss_idxs:
        faiss_score = faiss_dict[idx]
        bm25_score = bm25_norm[idx]
        score = bm25_weight * bm25_score + (1 - bm25_weight) * faiss_score
        combined_scores[idx] = score

    # Rank candidates
    ranked = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    ranked_idxs = [i for i, _ in ranked]

    # --- Apply optional filtering ---
    if seg_filter:
        filtered = [chunks[i] for i in ranked_idxs if seg_filter(chunks[i])]
        result = filtered[:k] + [chunks[i] for i in ranked_idxs if chunks[i] not in filtered][:max(0, k - len(filtered))]
    else:
        result = [chunks[i] for i in ranked_idxs[:k]]

    if preview:
        _print_preview(result, n_preview=500)

    return result