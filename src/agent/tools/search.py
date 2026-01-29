from typing import List
import faiss
from src.agent.types import ChunkMetadata
from src.retriever import _get_embedder

class IndexScout:
    """Semantic search returning structured metadata."""

    def __init__(self, faiss_index: faiss.Index, chunks: List[str], sources: List[str], embed_model: str):
        self.faiss_index = faiss_index
        self.chunks = chunks
        self.sources = sources
        self.embedder = _get_embedder(embed_model)

    def search(self, query: str, top_k: int = 10) -> List[ChunkMetadata]:
        q_vec = self.embedder.encode([query]).astype("float32")
        distances, indices = self.faiss_index.search(q_vec, top_k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            score = 1.0 / (1.0 + float(dist))
            preview = self.chunks[idx]
            results.append(
                ChunkMetadata(
                    chunk_id=int(idx),
                    score=score,
                    source=self.sources[idx] if idx < len(self.sources) else "unknown",
                    full_text=preview,
                )
            )
        return results

    def format_result(self, results: List[ChunkMetadata]) -> str:
        if not results:
            return "No results found."
        lines = ["Search results (use chunk_id with read_content):"]
        for i, r in enumerate(results):
            lines.append(f"  [{i}] chunk_id={r.chunk_id} score={r.score:.3f} source={r.source}")
            lines.append(f"      preview: {r.full_text}")
        return "\n".join(lines)
