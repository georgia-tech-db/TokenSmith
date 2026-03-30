from unittest.mock import patch

import networkx as nx
import pytest

from src.knowledge_graph.analysis import (
    analyze_query,
    compute_difficulty_features,
    compute_difficulty_score,
    extract_query_subgraph,
)
from src.knowledge_graph.models import (
    DifficultyCategory,
    QueryAnalysisResult,
    QueryFeatures,
)
from src.knowledge_graph.query import KGRetriever
from src.knowledge_graph.utils import KW_PATTERN, Normalizer, extract_ngrams


@pytest.fixture(scope="module")
def normalizer():
    return Normalizer()


@pytest.fixture
def linear_graph():
    """a -- b -- c"""
    g = nx.Graph()
    g.add_edge("a", "b")
    g.add_edge("b", "c")
    return g


@pytest.fixture
def kg_graph():
    """data --(w=2)-- structure --(w=1)-- algorithm"""
    g = nx.Graph()
    g.add_node("data", chunk_ids=[0, 1])
    g.add_node("structure", chunk_ids=[2])
    g.add_node("algorithm", chunk_ids=[3])
    g.add_edge("data", "structure", weight=2)
    g.add_edge("structure", "algorithm", weight=1)
    return g


@pytest.fixture
def kg_chunks():
    return {0: "text0", 1: "text1", 2: "text2", 3: "text3"}


class TestAnalysis:
    def test_extract_query_subgraph_includes_bridge(self, linear_graph):
        subg = extract_query_subgraph(["a", "c"], linear_graph)
        assert set(subg.nodes) == {"a", "b", "c"}

    def test_extract_query_subgraph_disconnected(self):
        g = nx.Graph()
        g.add_node("a")
        g.add_node("d")
        subg = extract_query_subgraph(["a", "d"], g)
        assert set(subg.nodes) == {"a", "d"}

    def test_extract_query_subgraph_single_node(self, linear_graph):
        subg = extract_query_subgraph(["a"], linear_graph)
        assert "a" in subg.nodes

    def test_compute_difficulty_score_easy(self):
        features = QueryFeatures(
            max_path_length=0, component_count=1, subgraph_node_count=5,
            avg_degree=1.0, doc_count=1,
        )
        result = compute_difficulty_score(features)
        assert result.score == 0
        assert result.category == DifficultyCategory.EASY

    def test_compute_difficulty_score_hard(self):
        features = QueryFeatures(
            max_path_length=3, component_count=3, subgraph_node_count=61,
            avg_degree=7.0, doc_count=5,
        )
        result = compute_difficulty_score(features)
        assert result.score == 10
        assert result.category == DifficultyCategory.HARD

    def test_compute_difficulty_score_medium_boundary(self):
        # multihop=1, fragmentation=1, subgraph_size=1, branching=1, dispersion=0 → total=4
        features = QueryFeatures(
            max_path_length=2, component_count=2, subgraph_node_count=21,
            avg_degree=4.0, doc_count=1,
        )
        result = compute_difficulty_score(features)
        assert result.score == 4
        assert result.category == DifficultyCategory.MEDIUM

    def test_compute_difficulty_score_components_populated(self):
        features = QueryFeatures(
            max_path_length=3, component_count=1, subgraph_node_count=5,
            avg_degree=1.0, doc_count=1,
        )
        result = compute_difficulty_score(features)
        assert result.components.multihop == 2
        assert result.components.fragmentation == 0

    def test_compute_difficulty_features_no_match(self, linear_graph):
        with patch("src.knowledge_graph.analysis.extract_query_nodes", return_value=[]):
            features = compute_difficulty_features("anything", linear_graph)
        assert features == QueryFeatures()

    def test_compute_difficulty_features_with_graph(self):
        g = nx.Graph()
        g.add_node("a", chunk_ids=[0])
        g.add_node("b", chunk_ids=[1])
        g.add_edge("a", "b", chunk_ids=[0, 1], weight=1)
        with patch("src.knowledge_graph.analysis.extract_query_nodes", return_value=["a", "b"]):
            features = compute_difficulty_features("a b", g)
        assert features.query_node_count == 2
        assert features.component_count == 1
        assert features.max_path_length == 1

    def test_analyze_query_returns_result(self, linear_graph):
        with patch("src.knowledge_graph.analysis.extract_query_nodes", return_value=["a"]):
            result = analyze_query("a", linear_graph)
        assert isinstance(result, QueryAnalysisResult)
        assert result.query == "a"
        assert result.features is not None
        assert result.difficulty is not None


class TestKGRetriever:
    def test_direct_match_scores_one(self, kg_graph, kg_chunks):
        retriever = KGRetriever(kg_graph, kg_chunks,
                                neighbor_weight=0.5, num_hops=1)
        with patch("src.knowledge_graph.query.extract_query_nodes", return_value=["data"]):
            results = retriever.retrieve_from_kg("data", top_k=10)
        scores = {cid: score for cid, _, score in results}
        assert scores[0] == pytest.approx(1.0)
        assert scores[1] == pytest.approx(1.0)
        # hop-1 neighbor "structure": 0.5 * (2/2) = 0.5
        assert scores[2] == pytest.approx(0.5)

    def test_no_match_returns_empty(self, kg_graph, kg_chunks):
        retriever = KGRetriever(kg_graph, kg_chunks)
        with patch("src.knowledge_graph.query.extract_query_nodes", return_value=[]):
            results = retriever.retrieve_from_kg("xyz", top_k=10)
        assert results == []

    def test_top_k_limits_results(self, kg_graph, kg_chunks):
        retriever = KGRetriever(kg_graph, kg_chunks, num_hops=1)
        with patch("src.knowledge_graph.query.extract_query_nodes", return_value=["data"]):
            results = retriever.retrieve_from_kg("data", top_k=1)
        assert len(results) == 1

    def test_results_sorted_descending(self, kg_graph, kg_chunks):
        retriever = KGRetriever(kg_graph, kg_chunks, num_hops=1)
        with patch("src.knowledge_graph.query.extract_query_nodes", return_value=["data"]):
            results = retriever.retrieve_from_kg("data", top_k=10)
        scores = [r[2] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_neighbor_hop_decay(self, kg_graph, kg_chunks):
        retriever = KGRetriever(kg_graph, kg_chunks,
                                neighbor_weight=0.5, num_hops=2)
        with patch("src.knowledge_graph.query.extract_query_nodes", return_value=["data"]):
            results = retriever.retrieve_from_kg("data", top_k=10)
        scores = {cid: score for cid, _, score in results}
        # hop-2: "algorithm" via "structure"; decay = 0.5^2 * (1/2) = 0.125
        assert scores[3] == pytest.approx(0.125)

    def test_chunk_text_in_results(self, kg_graph, kg_chunks):
        retriever = KGRetriever(kg_graph, kg_chunks, num_hops=0)
        with patch("src.knowledge_graph.query.extract_query_nodes", return_value=["data"]):
            results = retriever.retrieve_from_kg("data", top_k=10)
        for cid, text, _ in results:
            assert text == kg_chunks[cid]


class TestNormalizer:
    def test_lowercases(self, normalizer):
        result = normalizer.normalize(["Hello", "WORLD"])
        assert result == ["hello", "world"]

    def test_deduplication(self, normalizer):
        result = normalizer.normalize(["run", "run", "run"])
        assert result == ["run"]

    def test_empty_strings_skipped(self, normalizer):
        result = normalizer.normalize(["", "  ", "hello"])
        assert "hello" in result
        assert "" not in result

    def test_lemmatization(self, normalizer):
        result = normalizer.normalize(["running"])
        assert result == ["run"]

    def test_cross_form_deduplication(self, normalizer):
        # "run" and "running" both normalize to "run"
        result = normalizer.normalize(["run", "running", "database", "databases"])
        assert result == ["run", "database"]


class TestNgrams:
    EXPECTED_RESULTS = {
        'a data-structure algorithm', 'what is', 'data-structure algorithm', 'is a data-structure',
        'what', 'is', 'data-structure', 'a data-structure', 'is a', 'what is a', 'a', 'algorithm'
    }

    def test_bigrams_extracted(self):
        result = extract_ngrams("what is a data-structure algorithm?", KW_PATTERN)
        assert set(result) == self.EXPECTED_RESULTS
