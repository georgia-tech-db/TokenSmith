import logging
from itertools import combinations

import networkx as nx

from src.knowledge_graph.models import (
    DifficultyCategory,
    DifficultyComponents,
    DifficultyScore,
    QueryAnalysisResult,
    QueryFeatures,
)
from src.knowledge_graph.query import extract_query_nodes

logger = logging.getLogger(__name__)

# Scoring thresholds: [easy_max, medium_max] → scores [0, 1, 2]
# Each dimension contributes 0–2; total 0–10 maps to EASY/MEDIUM/HARD.
_MULTIHOP_THRESHOLDS = [1, 2]        # path hops: ≤1 direct, ≤2 one bridge, >2 multi-hop
_FRAGMENTATION_THRESHOLDS = [1, 2]   # components: 1 connected, 2 partly split, >2 fragmented
_SUBGRAPH_SIZE_THRESHOLDS = [20, 60] # subgraph nodes: small, moderate, large
_BRANCHING_THRESHOLDS = [3, 6]       # avg degree: low, moderate, high fan-out
_DISPERSION_THRESHOLDS = [2, 4]      # source docs: local, moderate, spread across many

# Simple heuristic thresholds for categorizing overall difficulty based on total score (0–10)
_CATEGORY_THRESHOLDS = [3, 7]        # total score: easy (≤3), medium (≤7), hard (>7)


def extract_query_subgraph(query_nodes: list[str], graph: nx.Graph) -> nx.Graph:
    """Return the subgraph spanning *query_nodes* and the shortest paths between them."""
    subgraph_nodes = set(query_nodes)
    for u, v in combinations(query_nodes, 2):
        if nx.has_path(graph, u, v):
            try:
                path = nx.shortest_path(graph, u, v)
                subgraph_nodes.update(path)
            except nx.NetworkXNoPath:
                pass
    return graph.subgraph(subgraph_nodes).copy()


def compute_difficulty_features(query: str, graph: nx.Graph) -> QueryFeatures:
    """Compute graph-structural features for *query*.

    Returns a zeroed ``QueryFeatures`` if no query nodes are found in *graph*.
    """
    query_nodes = extract_query_nodes(query, graph)
    logger.debug("Query nodes: %s", query_nodes)
    if not query_nodes:
        return QueryFeatures()

    subgraph = extract_query_subgraph(query_nodes, graph)

    component_count = nx.number_connected_components(subgraph)

    path_lengths = []
    for u, v in combinations(query_nodes, 2):
        if nx.has_path(graph, u, v):
            try:
                path_lengths.append(nx.shortest_path_length(graph, u, v))
            except nx.NetworkXNoPath:
                pass

    max_path_length = max(path_lengths) if path_lengths else 0
    avg_path_length = sum(path_lengths) / len(path_lengths) if path_lengths else 0.0

    degrees = dict(subgraph.degree())
    max_degree = max(degrees.values()) if degrees else 0
    avg_degree = sum(degrees.values()) / len(degrees) if degrees else 0.0

    chunk_ids: set[int] = set()
    for _, data in subgraph.nodes(data=True):
        chunk_ids.update(data.get("chunk_ids", []))
    for _, _, data in subgraph.edges(data=True):
        chunk_ids.update(data.get("chunk_ids", []))

    return QueryFeatures(
        query_node_count=len(query_nodes),
        component_count=component_count,
        max_path_length=max_path_length,
        avg_path_length=avg_path_length,
        avg_degree=avg_degree,
        max_degree=max_degree,
        subgraph_node_count=subgraph.number_of_nodes(),
        subgraph_edge_count=subgraph.number_of_edges(),
        doc_count=len(chunk_ids),
    )


def _map_to_score(
    value: int | float,
    thresholds: list[int | float],
    scores: list[int | DifficultyCategory],
):
    for threshold, score in zip(thresholds, scores):
        if value <= threshold:
            return score
    return scores[-1]


def compute_difficulty_score(features: QueryFeatures) -> DifficultyScore:
    multihop = _map_to_score(features.max_path_length, _MULTIHOP_THRESHOLDS, [0, 1, 2])
    fragmentation = _map_to_score(features.component_count, _FRAGMENTATION_THRESHOLDS, [0, 1, 2])
    subgraph_size = _map_to_score(features.subgraph_node_count, _SUBGRAPH_SIZE_THRESHOLDS, [0, 1, 2])
    branching = _map_to_score(features.avg_degree, _BRANCHING_THRESHOLDS, [0, 1, 2])
    dispersion = _map_to_score(features.doc_count, _DISPERSION_THRESHOLDS, [0, 1, 2])

    total = multihop + fragmentation + subgraph_size + branching + dispersion
    category = _map_to_score(
        total,
        _CATEGORY_THRESHOLDS,
        [DifficultyCategory.EASY, DifficultyCategory.MEDIUM, DifficultyCategory.HARD],
    )

    return DifficultyScore(
        score=total,
        category=category,
        components=DifficultyComponents(
            multihop=multihop,
            fragmentation=fragmentation,
            subgraph_size=subgraph_size,
            branching=branching,
            dispersion=dispersion,
        ),
    )


def analyze_query(query: str, graph: nx.Graph) -> QueryAnalysisResult:
    """Run the full difficulty analysis pipeline for *query*."""
    features = compute_difficulty_features(query, graph)
    difficulty = compute_difficulty_score(features)
    return QueryAnalysisResult(query=query, features=features, difficulty=difficulty)
