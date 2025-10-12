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


