from __future__ import annotations

from math import log2
from typing import Any, Dict, List, Optional, Sequence

from tests.metrics.base import MetricBase


def _sorted_gold_entries(retrieval_gold: Optional[Dict[str, Any]], key: str) -> List[Dict[str, int]]:
    if not retrieval_gold:
        return []
    entries = retrieval_gold.get(key, [])
    normalized = []
    for entry in entries:
        if key == "chunks":
            normalized.append({"id": int(entry["id"]), "grade": int(entry["grade"])})
        else:
            normalized.append({"page": int(entry["page"]), "grade": int(entry["grade"])})
    return normalized


def _retrieved_chunk_ids(actual_retrieved_chunks: Optional[Sequence[Dict[str, Any]]]) -> List[int]:
    if not actual_retrieved_chunks:
        return []
    return [int(chunk["chunk_id"]) for chunk in actual_retrieved_chunks]


def _retrieved_pages(actual_retrieved_chunks: Optional[Sequence[Dict[str, Any]]]) -> List[int]:
    if not actual_retrieved_chunks:
        return []
    pages = []
    for chunk in actual_retrieved_chunks:
        for page in chunk.get("page_numbers", []):
            page = int(page)
            if page not in pages:
                pages.append(page)
    return pages


def _dcg(grades: Sequence[int]) -> float:
    return sum((2**grade - 1) / log2(rank + 2) for rank, grade in enumerate(grades))


class RetrievalMetricBase(MetricBase):
    @property
    def weight(self) -> float:
        return 0.0

    @property
    def metric_group(self) -> str:
        return "retrieval"


class ChunkNDCGAt10Metric(RetrievalMetricBase):
    name = "chunk_ndcg_10"

    def calculate(
        self,
        retrieval_gold: Optional[Dict[str, Any]],
        actual_retrieved_chunks: Optional[Sequence[Dict[str, Any]]],
        **_: Any,
    ) -> float:
        chunk_grades = {entry["id"]: entry["grade"] for entry in _sorted_gold_entries(retrieval_gold, "chunks")}
        retrieved_ids = _retrieved_chunk_ids(actual_retrieved_chunks)[:10]
        observed_grades = [chunk_grades.get(chunk_id, 0) for chunk_id in retrieved_ids]
        ideal_grades = sorted(chunk_grades.values(), reverse=True)[:10]
        if not ideal_grades:
            return 0.0
        ideal_dcg = _dcg(ideal_grades)
        return _dcg(observed_grades) / ideal_dcg if ideal_dcg else 0.0


class ChunkRecallAt5Metric(RetrievalMetricBase):
    name = "chunk_recall_5"

    def calculate(self, retrieval_gold: Optional[Dict[str, Any]], actual_retrieved_chunks: Optional[Sequence[Dict[str, Any]]], **_: Any) -> float:
        gold_ids = {entry["id"] for entry in _sorted_gold_entries(retrieval_gold, "chunks") if entry["grade"] > 0}
        if not gold_ids:
            return 0.0
        retrieved_ids = set(_retrieved_chunk_ids(actual_retrieved_chunks)[:5])
        return len(gold_ids & retrieved_ids) / len(gold_ids)


class ChunkRecallAt10Metric(RetrievalMetricBase):
    name = "chunk_recall_10"

    def calculate(self, retrieval_gold: Optional[Dict[str, Any]], actual_retrieved_chunks: Optional[Sequence[Dict[str, Any]]], **_: Any) -> float:
        gold_ids = {entry["id"] for entry in _sorted_gold_entries(retrieval_gold, "chunks") if entry["grade"] > 0}
        if not gold_ids:
            return 0.0
        retrieved_ids = set(_retrieved_chunk_ids(actual_retrieved_chunks)[:10])
        return len(gold_ids & retrieved_ids) / len(gold_ids)


class ChunkMRRAt10Metric(RetrievalMetricBase):
    name = "chunk_mrr_10"

    def calculate(self, retrieval_gold: Optional[Dict[str, Any]], actual_retrieved_chunks: Optional[Sequence[Dict[str, Any]]], **_: Any) -> float:
        gold_ids = {entry["id"] for entry in _sorted_gold_entries(retrieval_gold, "chunks") if entry["grade"] > 0}
        for rank, chunk_id in enumerate(_retrieved_chunk_ids(actual_retrieved_chunks)[:10], start=1):
            if chunk_id in gold_ids:
                return 1.0 / rank
        return 0.0


class ChunkMAPAt10Metric(RetrievalMetricBase):
    name = "chunk_map_10"

    def calculate(self, retrieval_gold: Optional[Dict[str, Any]], actual_retrieved_chunks: Optional[Sequence[Dict[str, Any]]], **_: Any) -> float:
        gold_ids = {entry["id"] for entry in _sorted_gold_entries(retrieval_gold, "chunks") if entry["grade"] > 0}
        if not gold_ids:
            return 0.0

        hits = 0
        precision_sum = 0.0
        for rank, chunk_id in enumerate(_retrieved_chunk_ids(actual_retrieved_chunks)[:10], start=1):
            if chunk_id in gold_ids:
                hits += 1
                precision_sum += hits / rank
        return precision_sum / len(gold_ids)


class PageHitAt5Metric(RetrievalMetricBase):
    name = "page_hit_5"

    def calculate(self, retrieval_gold: Optional[Dict[str, Any]], actual_retrieved_chunks: Optional[Sequence[Dict[str, Any]]], **_: Any) -> float:
        gold_pages = {entry["page"] for entry in _sorted_gold_entries(retrieval_gold, "pages") if entry["grade"] > 0}
        if not gold_pages:
            return 0.0
        return float(any(page in gold_pages for page in _retrieved_pages(actual_retrieved_chunks)[:5]))


class PageHitAt10Metric(RetrievalMetricBase):
    name = "page_hit_10"

    def calculate(self, retrieval_gold: Optional[Dict[str, Any]], actual_retrieved_chunks: Optional[Sequence[Dict[str, Any]]], **_: Any) -> float:
        gold_pages = {entry["page"] for entry in _sorted_gold_entries(retrieval_gold, "pages") if entry["grade"] > 0}
        if not gold_pages:
            return 0.0
        return float(any(page in gold_pages for page in _retrieved_pages(actual_retrieved_chunks)[:10]))


class DirectPageHitAt10Metric(RetrievalMetricBase):
    name = "direct_page_hit_10"

    def calculate(self, retrieval_gold: Optional[Dict[str, Any]], actual_retrieved_chunks: Optional[Sequence[Dict[str, Any]]], **_: Any) -> float:
        gold_pages = {entry["page"] for entry in _sorted_gold_entries(retrieval_gold, "pages") if entry["grade"] == 3}
        if not gold_pages:
            return 0.0
        return float(any(page in gold_pages for page in _retrieved_pages(actual_retrieved_chunks)[:10]))
