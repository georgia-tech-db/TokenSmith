from src.planning.rules import (
    QUERY_TYPE_DEFINITION,
    QUERY_TYPE_EXPLANATORY,
    QUERY_TYPE_FOLLOW_UP,
    QUERY_TYPE_MULTI_PART,
    QUERY_TYPE_PROCEDURAL,
    classify_query,
    heuristic_decompose_query,
    should_apply_anchor_rerank,
)


def test_classify_query_uses_history_for_follow_up_detection():
    without_history = classify_query("Why is the last one important after a crash?", has_history=False)
    with_history = classify_query("Why is the last one important after a crash?", has_history=True)

    assert without_history.query_type == QUERY_TYPE_EXPLANATORY
    assert with_history.query_type == QUERY_TYPE_FOLLOW_UP
    assert "follow-up" in with_history.reason


def test_classify_query_routes_core_query_types():
    assert classify_query("What is transaction isolation?").query_type == QUERY_TYPE_DEFINITION
    assert classify_query("How does ARIES perform REDO?").query_type == QUERY_TYPE_PROCEDURAL
    assert classify_query("Explain why WAL matters.").query_type == QUERY_TYPE_EXPLANATORY
    assert classify_query("Compare primary keys and foreign keys.").query_type == QUERY_TYPE_MULTI_PART


def test_heuristic_decompose_query_splits_comparison_without_llm():
    sub_queries = heuristic_decompose_query("Compare primary keys and foreign keys.")

    assert sub_queries == [
        "What is primary keys?",
        "What is foreign keys?",
        "How do primary keys and foreign keys differ?",
    ]


def test_anchor_rerank_policy_skips_relationship_subqueries():
    assert should_apply_anchor_rerank("What problem does ARIES solve?")
    assert not should_apply_anchor_rerank("How do ARIES redo and undo differ?")
