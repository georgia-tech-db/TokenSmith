"""
reranker.py

This module supports re-ranking strategies applied before the generative LLM call.
"""

from typing import Dict, List, Tuple
from sentence_transformers import CrossEncoder

# -------------------------- Cross-Encoder Cache --------------------------
_CROSS_ENCODER_CACHE: Dict[str, CrossEncoder] = {}

def get_cross_encoder(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    """
    Fetch the cached cross-encoder model to prevent reloading on every query.
    """
    if model_name not in _CROSS_ENCODER_CACHE:
        print(f"Loading cross-encoder model: {model_name}...")
        _CROSS_ENCODER_CACHE[model_name] = CrossEncoder(model_name)
    return _CROSS_ENCODER_CACHE[model_name]


# -------------------------- Reranking Strategies -------------------------
def rerank_with_cross_encoder(
    query: str,
    candidates: List[Tuple[int, str]],
    top_n: int,
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> List[Tuple[int, float]]:
    """
    Reranks a list of (index, text) candidates using the cross-encoder model.
    Returns a list of (index, score) sorted descending by score.
    """
    if not candidates:
        return []

    model = get_cross_encoder(model_name)

    # Create pairs of [query, chunk] for the model
    pairs = [(query, chunk) for (_, chunk) in candidates]

    # Predict the scores
    scores = model.predict(pairs, show_progress_bar=False)

    # Combine indices with their scores and sort
    idx_with_scores = list(zip((idx for idx, _ in candidates), scores))
    idx_with_scores.sort(key=lambda x: x[1], reverse=True)

    return [(idx, float(score)) for idx, score in idx_with_scores[:top_n]]


# -------------------------- Reranking Router -----------------------------
def rerank(
    query: str,
    candidates: List[Tuple[int, str]],
    mode: str,
    top_n: int,
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> List[Tuple[int, float]]:
    """
    Routes to the appropriate reranker based on the mode in the config.
    """
    if mode == "cross_encoder":
        return rerank_with_cross_encoder(query, candidates, top_n, model_name=model_name)

    # We can add other re-ranking strategies in the future to switch between them.

    # Default is to do nothing (no-op), return indices with zero scores
    return [(idx, 0.0) for idx, _ in candidates][:top_n]
