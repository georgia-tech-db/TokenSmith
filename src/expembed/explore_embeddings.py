import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer

from src.config import QueryPlanConfig
from src.retriever import load_artifacts


def dump_chunks_to_file(chunks: list[str], output_file="all_chunks.txt"):
    """Writes all chunks with their indices to a text file."""
    print(f"Dumping {len(chunks)} chunks to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            f.write(f"--- CHUNK {i} ---\n{chunk.strip()}\n\n")
    print("Dump complete.")


def visualize_embeddings(index, embedder: SentenceTransformer, query: str | None = None):
    """Creates and displays a PCA plot of all embeddings in the index."""
    all_embeddings = index.reconstruct_n(0, index.ntotal)
    labels = [str(i) for i in range(index.ntotal)]

    if query:
        query_embedding = embedder.encode([query])
        all_embeddings = np.vstack([query_embedding, all_embeddings])
        labels.insert(0, "QUERY")

    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(all_embeddings)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(16, 12))

    # Plot all chunk points
    chunk_data = embeddings_2d[1:] if query else embeddings_2d
    ax.scatter(chunk_data[:, 0], chunk_data[:, 1], alpha=0.5, label="Chunks")

    # Label points with their chunk IDs
    for i in range(chunk_data.shape[0]):
        ax.text(chunk_data[i, 0], chunk_data[i, 1], str(i), fontsize=8, alpha=0.7)

    if query:
        ax.scatter(embeddings_2d[0, 0], embeddings_2d[0, 1], c='red', marker='*', s=200, label="Query")
        ax.text(embeddings_2d[0, 0], embeddings_2d[0, 1], "QUERY", fontsize=12, c='red', ha='right')

    ax.set_title(f"PCA of All Chunk Embeddings")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.legend()
    plt.tight_layout()
    plt.show()


def main():
    cfg = QueryPlanConfig.from_yaml("config/config.yaml")
    index, chunks, _, _, _ = load_artifacts(cfg.index_prefix, cfg)
    embedder = SentenceTransformer(cfg.embed_model)
    dump_chunks_to_file(chunks)
    test_query = "Compare the Range-Partitioning Sort and Parallel External Sort-Merge algorithms."
    visualize_embeddings(index, embedder, query=test_query)


if __name__ == "__main__":
    main()