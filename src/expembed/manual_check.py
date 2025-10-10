import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import QueryPlanConfig
from src.retriever import load_artifacts, get_candidates


def main():
    """
    Runs a targeted check on embedding similarity and index recall
    for a given query and a manually-defined set of relevant chunks.
    """
    cfg = QueryPlanConfig.from_yaml("config/config.yaml")
    index, chunks, _, _, _ = load_artifacts(cfg.index_prefix, cfg)
    embedder = SentenceTransformer(cfg.embed_model)

    # --- Define the "Golden Set" ---
    test_query = "Compare the Range-Partitioning Sort and Parallel External Sort-Merge algorithms."
    manual_relevant_indices = [4, 5, 6, 1, 2, 3, 29, 16, 89, 0]

    print(f"\nQUERY: '{test_query}'")
    print(f"MANUAL INDICES: {manual_relevant_indices}")

    # --- 1. Check Embedding Similarity ---
    print("\n--- Embedding Similarity (Query vs. Manual Chunks) ---")
    query_embedding = embedder.encode([test_query])
    relevant_chunk_embeddings = embedder.encode([chunks[i] for i in manual_relevant_indices])
    similarities = cosine_similarity(query_embedding, relevant_chunk_embeddings)[0]

    for i, chunk_idx in enumerate(manual_relevant_indices):
        print(f"Chunk {chunk_idx:<4} | Cosine Similarity: {similarities[i]:.4f}")

    # --- 2. Check Vector Index Retrieval ---
    print("\n--- Vector Index Retrieval (FAISS Search) ---")
    retrieved_indices, _ = get_candidates(test_query, 20, index, chunks, embed_model=cfg.embed_model)
    hits = set(manual_relevant_indices) & set(retrieved_indices)

    print(f"Top 20 retrieved by FAISS: {retrieved_indices}")
    print(
        f"Recall@20: {len(hits) / len(manual_relevant_indices):.2%} ({len(hits)} of {len(manual_relevant_indices)} manual chunks found)")


if __name__ == "__main__":
    main()