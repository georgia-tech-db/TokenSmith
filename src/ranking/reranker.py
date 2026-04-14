"""
reranker.py

This module supports re-ranking strategies applied before the generative LLM call.
"""

from typing import Dict, List, Tuple
from sentence_transformers import CrossEncoder

# -------------------------- Cross-Encoder Cache --------------------------
_CROSS_ENCODER_CACHE: Dict[str, CrossEncoder] = {}

def get_cross_encoder(model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2"):
    """Fetch the cached cross-encoder model to prevent reloading on every query."""
    if model_name not in _CROSS_ENCODER_CACHE:
        _CROSS_ENCODER_CACHE[model_name] = CrossEncoder(model_name)
    return _CROSS_ENCODER_CACHE[model_name]


# -------------------------- Reranking Strategies -------------------------

def rerank_with_cross_encoder(
    query: str,
    indexed_chunks: List[Tuple[int, str]],
    top_n: int,
) -> List[Tuple[int, str, float]]:
    """
    Reranks a list of (idx, text) pairs using the cross-encoder model.

    Args:
        query:          The user query.
        indexed_chunks: List of (original_chunk_idx, chunk_text) pairs.
        top_n:          Number of top results to return.

    Returns:
        List of (original_chunk_idx, chunk_text, cross_encoder_score),
        sorted by cross-encoder score descending, truncated to top_n.
    """
    if not indexed_chunks:
        return []

    model = get_cross_encoder()

    pairs = [(query, chunk_text) for _, chunk_text in indexed_chunks]
    scores = model.predict(pairs, show_progress_bar=False)

    scored = [
        (idx, chunk_text, float(score))
        for (idx, chunk_text), score in zip(indexed_chunks, scores)
    ]
    scored.sort(key=lambda x: x[2], reverse=True)

    return scored[:top_n]


# -------------------------- Reranking Router -----------------------------

def rerank(
    query: str,
    indexed_chunks: List[Tuple[int, str]],
    mode: str,
    top_n: int,
) -> List[Tuple[int, str, float]]:
    """
    Routes to the appropriate reranker based on mode.

    Args:
        query:          The user query.
        indexed_chunks: List of (original_chunk_idx, chunk_text) pairs.
        mode:           Reranking mode, e.g. 'cross_encoder'.
        top_n:          Number of top results to return.

    Returns:
        List of (original_chunk_idx, chunk_text, cross_encoder_score).
        If no reranking mode matches, returns input as (idx, text, 0.0) tuples.
    """
    if mode == "cross_encoder":
        return rerank_with_cross_encoder(query, indexed_chunks, top_n)

    # No reranking — return with a placeholder score of 0.0
    return [(idx, text, 0.0) for idx, text in indexed_chunks]