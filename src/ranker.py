"""
HALO-style Validate re-ranker (stubbed pass-through unless mode == "halo").
You can slot in full HALO logic later without touching main.py.
"""
from typing import List

try:
    # optional import—only needed if you actually wire HALO
    from src.pipeline.halo_pipeline import HALOPipeline
except ImportError:
    HALOPipeline = None

def rerank(query: str, chunks: List[str], mode: str = "none") -> List[str]:
    if mode != "halo" or HALOPipeline is None:
        return chunks  # No-op
    # HALO expects dataset-style dicts; wrap minimal stubs
    halo_input = [{"Question": query, "Context": "\n".join(chunks)}]
    pipeline   = HALOPipeline()
    best_order = pipeline.run_single(query, chunks)  # pseudo-API
    return best_order
