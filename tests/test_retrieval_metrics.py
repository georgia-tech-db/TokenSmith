import math

import pytest

from tests.metrics.chunk_retrieval import (
    ChunkMAPAt10Metric,
    ChunkMRRAt10Metric,
    ChunkNDCGAt10Metric,
    ChunkRecallAt5Metric,
    ChunkRecallAt10Metric,
    DirectPageHitAt10Metric,
    PageHitAt5Metric,
    PageHitAt10Metric,
)


pytestmark = pytest.mark.unit


@pytest.fixture
def retrieval_gold():
    return {
        "chunks": [
            {"id": 10, "grade": 3},
            {"id": 11, "grade": 2},
            {"id": 12, "grade": 1},
        ],
        "pages": [
            {"page": 100, "grade": 3},
            {"page": 101, "grade": 2},
        ],
    }


@pytest.fixture
def actual_chunks():
    return [
        {"chunk_id": 11, "page_numbers": [101]},
        {"chunk_id": 99, "page_numbers": [250]},
        {"chunk_id": 10, "page_numbers": [100]},
        {"chunk_id": 12, "page_numbers": [102]},
    ]


def test_chunk_ndcg_10_uses_graded_relevance(retrieval_gold, actual_chunks):
    metric = ChunkNDCGAt10Metric()
    score = metric.calculate(retrieval_gold=retrieval_gold, actual_retrieved_chunks=actual_chunks)
    assert 0 < score < 1
    assert math.isclose(score, 0.73787, rel_tol=1e-4)


def test_chunk_recall_metrics(retrieval_gold, actual_chunks):
    assert ChunkRecallAt5Metric().calculate(retrieval_gold=retrieval_gold, actual_retrieved_chunks=actual_chunks) == 1.0
    assert ChunkRecallAt10Metric().calculate(retrieval_gold=retrieval_gold, actual_retrieved_chunks=actual_chunks) == 1.0


def test_chunk_mrr_and_map(retrieval_gold, actual_chunks):
    assert math.isclose(
        ChunkMRRAt10Metric().calculate(retrieval_gold=retrieval_gold, actual_retrieved_chunks=actual_chunks),
        1.0,
    )
    assert math.isclose(
        ChunkMAPAt10Metric().calculate(retrieval_gold=retrieval_gold, actual_retrieved_chunks=actual_chunks),
        0.8055555,
        rel_tol=1e-6,
    )


def test_page_hit_metrics(retrieval_gold, actual_chunks):
    assert PageHitAt5Metric().calculate(retrieval_gold=retrieval_gold, actual_retrieved_chunks=actual_chunks) == 1.0
    assert PageHitAt10Metric().calculate(retrieval_gold=retrieval_gold, actual_retrieved_chunks=actual_chunks) == 1.0
    assert DirectPageHitAt10Metric().calculate(retrieval_gold=retrieval_gold, actual_retrieved_chunks=actual_chunks) == 1.0
