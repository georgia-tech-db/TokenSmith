"""
reranker.py

This module supports re-ranking strategies applied before the generative LLM call.
"""

from typing import Dict, List
from sentence_transformers import CrossEncoder

# -------------------------- Cross-Encoder Cache --------------------------
_CROSS_ENCODER_CACHE: Dict[str, CrossEncoder] = {}

def get_cross_encoder(model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2"):
    """
    Fetch the cached cross-encoder model to prevent reloading on every query.
    """
    if model_name not in _CROSS_ENCODER_CACHE:
        _CROSS_ENCODER_CACHE[model_name] = CrossEncoder(model_name)
    return _CROSS_ENCODER_CACHE[model_name]


# -------------------------- Reranking Strategies -------------------------
def rerank_with_cross_encoder(query: str, chunks: List[str], top_n: int) -> List[str]:
    """
    Reranks a list of documents using the cross-encoder model.
    """
    if not chunks:
        return []

    model = get_cross_encoder()

    # Create pairs of [query, chunk] for the model
    pairs = [(query, chunk) for chunk in chunks]

    # Predict the scores
    scores = model.predict(pairs, show_progress_bar=False)

    # Combine chunks with their scores and sort
    chunk_with_scores = list(zip(chunks, scores))
    chunk_with_scores.sort(key=lambda x: x[1], reverse=True)

    reordered_chunks = []

    for chunk, score in chunk_with_scores:
        # Only include chunks with positive scores
        if score > 0:
            reordered_chunks.append(chunk)

    # Return top N chunks
    return reordered_chunks[0:top_n]


# -------------------------- Reranking Router -----------------------------
def rerank(query: str, chunks: List[str], mode: str, top_n: int) -> List[str]:
    """
    Routes to the appropriate reranker based on the mode in the config.
    """
    if mode == "cross_encoder":
        return rerank_with_cross_encoder(query, chunks, top_n)

    # We can add other re-ranking strategies to switch between them.
    return chunks