import pytest

from src.artifacts import ArtifactBundle
from src.config import RAGConfig
from src.retrieval_pipeline import (
    AdaptiveQueryPlanner,
    _diversify_by_section,
    _page_aware_rerank,
    execute_retrieval_plan,
)


pytestmark = pytest.mark.unit


class StubRetriever:
    def __init__(self, name, score_map):
        self.name = name
        self.score_map = score_map
        self.calls = []

    def get_scores(self, query, pool_size, texts, candidate_ids=None):
        self.calls.append(
            {
                "query": query,
                "candidate_ids": None if candidate_ids is None else sorted(int(idx) for idx in candidate_ids),
            }
        )
        scores = dict(self.score_map.get(query, {}))
        if candidate_ids is not None:
            allowed_ids = {int(idx) for idx in candidate_ids}
            scores = {idx: score for idx, score in scores.items() if idx in allowed_ids}
        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:pool_size]
        return dict(ordered)


def _build_bundle(*, chunks, metadata, sections=None, section_meta=None):
    return ArtifactBundle(
        chunk_index=None,
        chunk_bm25=None,
        chunks=chunks,
        sources=["textbook"] * len(chunks),
        metadata=metadata,
        page_to_chunk_map={},
        section_index=object() if sections else None,
        section_bm25=None,
        sections=sections or [],
        section_sources=["textbook"] * len(sections or []),
        section_meta=section_meta or [],
    )


def test_adaptive_planner_classifies_follow_up_and_rewrites_route():
    cfg = RAGConfig()
    planner = AdaptiveQueryPlanner(cfg)
    query_type, reason = planner.classify(
        "Why is that?",
        history=[{"role": "user", "content": "What is ARIES?"}],
    )
    assert query_type == "follow_up"
    assert "follow-up" in reason


def test_adaptive_planner_classifies_multi_part():
    cfg = RAGConfig()
    planner = AdaptiveQueryPlanner(cfg)
    query_type, _ = planner.classify("Compare atomicity and durability, and explain why both matter.")
    assert query_type == "multi_part"


def test_adaptive_planner_heuristically_decomposes_comparison_query():
    cfg = RAGConfig()
    planner = AdaptiveQueryPlanner(cfg)
    plan, _ = planner.plan("Compare ARIES redo and undo.")
    assert plan.retrieval_mode == "hierarchical"
    assert "What is ARIES redo?" in plan.sub_queries
    assert "What is undo.?" not in plan.sub_queries
    assert any("differ" in sub_query.lower() for sub_query in plan.sub_queries)


def test_adaptive_planner_uses_llm_decomposition_when_heuristics_do_not_split(monkeypatch):
    monkeypatch.setattr(
        "src.retrieval_pipeline.decompose_complex_query",
        lambda query, model_path: [
            "What does the ARIES analysis pass do?",
            "How does the ARIES redo pass work?",
            "When does ARIES perform undo?",
        ],
    )
    cfg = RAGConfig()
    planner = AdaptiveQueryPlanner(cfg)
    plan, _ = planner.plan("Explain ARIES analysis, redo, and undo.")
    assert plan.query_type == "multi_part"
    assert plan.retrieval_mode == "hierarchical"
    assert plan.sub_queries == [
        "What does the ARIES analysis pass do?",
        "How does the ARIES redo pass work?",
        "When does ARIES perform undo?",
        "Explain ARIES analysis, redo, and undo.",
    ]


def test_adaptive_planner_rewrites_follow_up_into_hierarchical_query(monkeypatch):
    monkeypatch.setattr(
        "src.retrieval_pipeline.contextualize_query",
        lambda query, history, model_path: "Why does ARIES perform undo after redo?",
    )
    cfg = RAGConfig()
    planner = AdaptiveQueryPlanner(cfg)
    plan, _ = planner.plan(
        "Why is that?",
        history=[{"role": "user", "content": "How does the ARIES redo pass work?"}],
    )
    assert plan.query_type == "follow_up"
    assert plan.resolved_query_type == "explanatory"
    assert plan.rewritten_query == "Why does ARIES perform undo after redo?"
    assert plan.retrieval_mode == "hierarchical"


def test_page_aware_rerank_prefers_tighter_direct_page_spans():
    reranked = _page_aware_rerank(
        query="What is ARIES redo?",
        candidate_ids=[5, 9],
        candidate_scores=[0.95, 0.92],
        metadata=[
            {},
            {},
            {},
            {},
            {},
            {"page_numbers": [918, 919, 920], "raw_text": "ARIES redo repeats history across multiple pages."},
            {},
            {},
            {},
            {"page_numbers": [918], "raw_text": "The redo pass repeats history by replaying actions from RedoLSN."},
        ],
        chunks=[""] * 10,
    )
    assert reranked[0] == 9


def test_diversify_by_section_limits_duplicate_section_selections():
    diversified = _diversify_by_section(
        candidate_ids=[0, 1, 2, 3],
        metadata=[
            {"section_id": 7},
            {"section_id": 7},
            {"section_id": 8},
            {"section_id": 8},
        ],
        limit=3,
        max_per_section=1,
    )
    assert diversified == [0, 2, 1]


def test_execute_retrieval_plan_uses_section_stage_to_gate_chunk_candidates():
    cfg = RAGConfig(top_k=2, num_candidates=4, section_top_k=1, page_rerank_window=4)
    bundle = _build_bundle(
        chunks=[
            "ARIES redo repeats history from RedoLSN.",
            "Redo starts from the smallest recLSN in the dirty-page table.",
            "Undo rolls back incomplete transactions.",
        ],
        metadata=[
            {"section_id": 0, "page_numbers": [1463], "raw_text": "ARIES redo repeats history from RedoLSN."},
            {"section_id": 0, "page_numbers": [1462], "raw_text": "Redo starts from the smallest recLSN in the dirty-page table."},
            {"section_id": 1, "page_numbers": [1464], "raw_text": "Undo rolls back incomplete transactions."},
        ],
        sections=["ARIES recovery overview", "ARIES undo"],
        section_meta=[
            {"section_id": 0, "section_path": "Chapter 19 Recovery ARIES", "chunk_ids": [0, 1]},
            {"section_id": 1, "section_path": "Chapter 19 Recovery ARIES Undo", "chunk_ids": [2]},
        ],
    )
    chunk_retriever = StubRetriever(
        "faiss",
        {
            "Explain how the ARIES redo pass works.": {
                2: 0.99,
                0: 0.92,
                1: 0.88,
            }
        },
    )
    section_retriever = StubRetriever(
        "faiss",
        {
            "Explain how the ARIES redo pass works.": {
                0: 1.0,
            }
        },
    )

    _, final_chunk_ids, trace = execute_retrieval_plan(
        query="Explain how the ARIES redo pass works.",
        cfg=cfg,
        bundle=bundle,
        retrievers={"chunk": [chunk_retriever], "section": [section_retriever]},
        history=[],
    )

    assert trace.retrieval_mode == "hierarchical"
    assert trace.fused_section_ids == [0]
    assert chunk_retriever.calls[0]["candidate_ids"] == [0, 1]
    assert final_chunk_ids == [0, 1]
    assert trace.selected_section_paths == ["Chapter 19 Recovery ARIES"]


def test_execute_retrieval_plan_merges_multi_part_results_with_coverage_bonus():
    cfg = RAGConfig(top_k=2, num_candidates=5, page_rerank_window=5)
    bundle = _build_bundle(
        chunks=[
            "ARIES redo replays actions forward from RedoLSN.",
            "ARIES undo rolls back incomplete transactions using compensation log records.",
            "Redo and undo differ because redo repeats history while undo reverses incomplete work.",
        ],
        metadata=[
            {"section_id": 0, "page_numbers": [1463], "raw_text": "ARIES redo replays actions forward from RedoLSN."},
            {"section_id": 1, "page_numbers": [1463], "raw_text": "ARIES undo rolls back incomplete transactions using compensation log records."},
            {"section_id": 2, "page_numbers": [1463], "raw_text": "Redo and undo differ because redo repeats history while undo reverses incomplete work."},
        ],
    )
    chunk_retriever = StubRetriever(
        "faiss",
        {
            "What is ARIES redo?": {0: 1.0, 2: 0.72},
            "What is undo?": {1: 1.0, 2: 0.71},
            "How do ARIES redo and undo differ?": {2: 1.0, 0: 0.4, 1: 0.4},
        },
    )

    _, final_chunk_ids, trace = execute_retrieval_plan(
        query="Compare ARIES redo and undo.",
        cfg=cfg,
        bundle=bundle,
        retrievers={"chunk": [chunk_retriever], "section": []},
        history=[],
    )

    assert trace.query_type == "multi_part"
    assert len(trace.subquery_traces) == len(trace.sub_queries)
    assert len(trace.subquery_traces) >= 3
    assert final_chunk_ids[0] == 2
    assert any("differ" in subquery_trace["query"].lower() for subquery_trace in trace.subquery_traces)
