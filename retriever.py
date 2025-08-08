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
    seg_filter: Optional[Callable[[str], bool]] = None,
    preview: bool = True,
) -> List[str]:
    """
    Return up to k chunks for the query.

    Strategy:
      1) Search FAISS over the full corpus to get a ranked candidate list.
      2) If seg_filter is provided, keep only candidates where seg_filter(chunk) == True.
         If fewer than k survive, fall back to unfiltered to fill.
    """
    # 1) FAISS search
    embedder = _get_embedder(embed_model)
    q_vec = embedder.encode([query]).astype("float32")
    # Fetch more than k to have room for filtering
    overshoot = max(k * 3, k + 5)
    D, I = index.search(q_vec, overshoot)
    ranked_idxs = I[0].tolist()

    # Guard against out-of-range indices (in case of any mismatch)
    ranked_idxs = [i for i in ranked_idxs if 0 <= i < len(chunks)]

    # 2) Apply optional Segment/Filter post-hoc
    if seg_filter:
        filtered = [chunks[i] for i in ranked_idxs if seg_filter(chunks[i])]
        if len(filtered) >= k:
            result = filtered[:k]
        else:
            # fill remaining from the original ranked list
            need = k - len(filtered)
            fill = [chunks[i] for i in ranked_idxs if chunks[i] not in filtered][:need]
            result = filtered + fill
    else:
        result = [chunks[i] for i in ranked_idxs[:k]]

    if preview:
        _print_preview(result, n_preview=500)

    return result
