from typing import List, Tuple

class NavigationalReader:
    """Read chunks with relative offset navigation."""

    def __init__(self, chunks: List[str], sources: List[str]):
        self.chunks = chunks
        self.sources = sources

    def read(self, target_chunk_id: int, relative_start: int = 0, relative_end: int = 0) -> Tuple[str, List[int]]:
        start_idx = max(0, target_chunk_id + relative_start)
        end_idx = min(len(self.chunks), target_chunk_id + relative_end + 1)

        if start_idx >= len(self.chunks) or end_idx <= start_idx:
            return "", []

        chunk_ids = list(range(start_idx, end_idx))
        texts = []
        for cid in chunk_ids:
            src = self.sources[cid] if cid < len(self.sources) else "unknown"
            texts.append(f"--- Chunk {cid} (source: {src}) ---\n{self.chunks[cid]}")

        return "\n\n".join(texts), chunk_ids

    def format_result(self, text: str, chunk_ids: List[int]) -> str:
        if not text:
            return "No content found for specified range."
        return f"Content from chunks {chunk_ids}:\n\n{text}"
