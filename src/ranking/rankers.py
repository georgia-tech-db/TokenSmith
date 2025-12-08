from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Dict, Any

from rank_bm25 import BM25Okapi
from src.ranking.tagging import query_top_tags, tag_affinity_score

# typedef Candidate as base, we might change this into a class later
# Each candidate is identified by its global index into `chunks`
Candidate = int


class Ranker(ABC):
    name: str

    @abstractmethod
    def prepare(self, *, query: str, chunks: List[str], cand_idxs: List[Candidate], context: Dict[str, Any]) -> None:
        """Optional precomputation per query/pool."""

    @abstractmethod
    def score(self, *, query: str, chunks: List[str], cand_idxs: List[Candidate], context: Dict[str, Any]) -> Dict[
        Candidate, float]:
        """Return raw, comparable scores for each candidate (higher is better)."""

class FaissSimilarityRanker(Ranker):
    name = "faiss"

    def prepare(self, *, query, chunks, cand_idxs, context):
        pass

    def score(self, *, query, chunks, cand_idxs, context) -> Dict[int, float]:
        # expects context["faiss_distances"]: Dict[idx, distance]
        dists = context.get("faiss_distances", {})
        # convert L2 distance to similarity scores
        sims = {i: 1.0 / (1.0 + dists.get(i, 1e6)) for i in cand_idxs}
        """
        faiss_sims = 1.0 / (1.0 + D[0][: len(cand_idxs)])
        faiss_norm = (faiss_sims - np.min(faiss_sims)) / (np.ptp(faiss_sims) + 1e-8)
        faiss_score: Dict[int, float] = {idx: float(s) for idx, s in zip(cand_idxs, faiss_norm)}
        """
        return sims


class BM25Ranker(Ranker):
    name = "bm25"

    def prepare(self, *, query, chunks, cand_idxs, context):
        docs = [chunks[i].lower().split() for i in cand_idxs]
        context["_bm25_docs"] = docs
        context["_bm25"] = BM25Okapi(docs)

    def score(self, *, query, chunks, cand_idxs, context):
        bm = context.get("_bm25")
        if not bm: return {i: 0.0 for i in cand_idxs}
        toks = query.lower().split()
        vals = bm.get_scores(toks)
        return {i: float(v) for i, v in zip(cand_idxs, vals)}


class TfIDFRanker(Ranker):
    name = "tf-idf"

    def prepare(self, *, query, chunks, cand_idxs, context):
        vec = context.get("vectorizer")
        context["_q_tags"] = query_top_tags(query, vec, top_q=8) if vec else []

    def score(self, *, query, chunks, cand_idxs, context):
        qtags = context.get("_q_tags", [])
        chunk_tags = context.get("chunk_tags", [])
        out = {}
        for i in cand_idxs:
            tags_i = chunk_tags[i] if i < len(chunk_tags) and chunk_tags else []
            out[i] = tag_affinity_score(tags_i, qtags, mode="weighted", tag_weights=None)
        return out


class LocationRanker(Ranker):
    name = "location"
    
    def prepare(self, *, query, chunks, cand_idxs, context):
        """Extract location hints from query and prepare for scoring."""
        # Location hints are passed via context from the query planner
        context["_location_hint"] = context.get("location_hint", None)
    
    def score(self, *, query, chunks, cand_idxs, context):
        """Score chunks based on location hints (chapter/section matching)."""
        hint = context.get("_location_hint")
        if not hint:
            return {i: 0.0 for i in cand_idxs}
        
        want_ch = hint.get("chapter") if isinstance(hint, dict) else None
        want_sec = hint.get("section") if isinstance(hint, dict) else None
        metadata = context.get("metadata", [])
        
        out = {}
        for i in cand_idxs:
            score = 0.0
            try:
                section_heading = metadata[i].get("section") if i < len(metadata) else None
            except Exception:
                section_heading = None
            
            if section_heading:
                ch, sec = _extract_numbering_from_heading(section_heading)
                
                # Section match gets higher score (more specific)
                if want_sec and sec and str(sec).startswith(str(want_sec)):
                    score += 1.0
                
                # Chapter match gets lower score (less specific)
                if want_ch is not None and ch is not None and int(ch) == int(want_ch):
                    score += 0.5
            
            out[i] = score
        
        return out


def _extract_numbering_from_heading(heading: str) -> tuple[Optional[int], Optional[str]]:
    """
    Extract chapter and section numbers from a heading string.
    Returns (chapter_num, section_str) or (None, None) if not found.
    """
    if not heading:
        return None, None
    
    import re
    
    # Try to match section patterns like "19.3", "19.3.1", etc.
    section_match = re.search(r'(\d+(?:\.\d+)+)', heading)
    if section_match:
        section_str = section_match.group(1)
        # Extract chapter from section (first number)
        chapter_match = re.search(r'^(\d+)', section_str)
        chapter_num = int(chapter_match.group(1)) if chapter_match else None
        return chapter_num, section_str
    
    # Try to match chapter patterns like "Chapter 19", "Ch. 19", etc.
    chapter_match = re.search(r'(?:chapter|ch\.?)\s*(\d+)', heading, re.IGNORECASE)
    if chapter_match:
        chapter_num = int(chapter_match.group(1))
        return chapter_num, None
    
    return None, None


