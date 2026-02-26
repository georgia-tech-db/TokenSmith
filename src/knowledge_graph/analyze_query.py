import json
import networkx as nx
from itertools import combinations
import argparse
from nltk.util import ngrams
from models import (
    QueryFeatures,
    DifficultyCategory,
    DifficultyScore,
    QueryAnalysisResult,
    DifficultyComponents,
)


def load_graph(filepath: str) -> nx.Graph:
    with open(filepath, "r") as f:
        data = json.load(f)
    return nx.node_link_graph(data)


def extract_query_nodes(query: str, graph: nx.Graph) -> list[str]:
    split_query = query.lower().removesuffix("?").removesuffix(".").split()
    bigrams = list(ngrams(split_query, 2))
    matched_nodes: set[str] = set()

    for bigram in bigrams:
        composite = " ".join(bigram)
        if graph.has_node(composite):
            matched_nodes.add(composite)

    for word in split_query:
        if graph.has_node(word):
            matched_nodes.add(word)
    return list(matched_nodes)


def extract_query_subgraph(query_nodes: list[str], graph: nx.Graph) -> nx.Graph:
    subgraph_nodes = set(query_nodes)
    for u, v in combinations(query_nodes, 2):
        if nx.has_path(graph, u, v):
            try:
                # Get the shortest path
                path = nx.shortest_path(graph, u, v)
                # Add all nodes in the path to the subgraph (todo: we can limit the path length)
                subgraph_nodes.update(path)
            except nx.NetworkXNoPath:
                pass
    return graph.subgraph(subgraph_nodes).copy()


def compute_difficulty_features(query: str, graph: nx.Graph):
    query_nodes = extract_query_nodes(query, graph)
    # Number of query subgraph nodes
    E_q = len(query_nodes)
    print(query_nodes)
    if E_q == 0:
        return QueryFeatures()

    subgraph = extract_query_subgraph(query_nodes, graph)

    # Number of connected components in the query subgraph
    C = nx.number_connected_components(subgraph) if len(subgraph) > 0 else 0

    # Store path lengths between query nodes
    path_lengths = []
    for u, v in combinations(query_nodes, 2):
        if nx.has_path(graph, u, v):
            try:
                length = nx.shortest_path_length(graph, u, v)
                path_lengths.append(length)
            except nx.NetworkXNoPath:
                pass
    # Max path length between query nodes
    L_max = max(path_lengths) if path_lengths else 0
    # Average path length between query nodes
    L_avg = sum(path_lengths) / len(path_lengths) if path_lengths else 0.0

    degrees = dict(subgraph.degree())
    # Max degree of nodes in the query subgraph
    D_max = max(degrees.values()) if degrees else 0
    # Average degree of nodes in the query subgraph
    D_avg = sum(degrees.values()) / len(degrees) if degrees else 0.0

    # Number of nodes in the query subgraph
    N_sub = subgraph.number_of_nodes()
    # Number of edges in the query subgraph
    M_sub = subgraph.number_of_edges()

    # Retrieval dispersion: distinct source documents in subgraph
    chunk_ids = set()
    for node, data in subgraph.nodes(data=True):
        for cid in data.get("chunk_ids", []):
            chunk_ids.add(cid)
    for u, v, data in subgraph.edges(data=True):
        for cid in data.get("chunk_ids", []):
            chunk_ids.add(cid)

    Doc_count = len(chunk_ids)

    return QueryFeatures(
        E_q=E_q,
        C=C,
        L_max=L_max,
        L_avg=L_avg,
        D_avg=D_avg,
        D_max=D_max,
        N_sub=N_sub,
        M_sub=M_sub,
        Doc_count=Doc_count,
    )


def compute_difficulty_score(features: QueryFeatures) -> DifficultyScore:
    # 1. Multi-hop complexity
    L_max = features.L_max
    if L_max <= 1:
        s1 = 0
    elif L_max == 2:
        s1 = 1
    else:
        s1 = 2

    # 2. Graph fragmentation
    C = features.C
    if C <= 1:
        s2 = 0
    elif C == 2:
        s2 = 1
    else:
        s2 = 2

    # 3. Subgraph size
    N_sub = features.N_sub
    if N_sub <= 20:
        s3 = 0
    elif N_sub <= 60:
        s3 = 1
    else:
        s3 = 2

    # 4. Branching
    D_avg = features.D_avg
    if D_avg <= 3:
        s4 = 0
    elif D_avg <= 6:
        s4 = 1
    else:
        s4 = 2

    # 5. Dispersion
    Doc_count = features.Doc_count
    if Doc_count <= 2:
        s5 = 0
    elif Doc_count <= 4:
        s5 = 1
    else:
        s5 = 2

    diff = s1 + s2 + s3 + s4 + s5

    if diff <= 2:
        category = DifficultyCategory.EASY
    elif diff <= 5:
        category = DifficultyCategory.MEDIUM
    else:
        category = DifficultyCategory.HARD

    return DifficultyScore(
        score=diff,
        category=category,
        components=DifficultyComponents(
            s1_multihop=s1,
            s2_fragmentation=s2,
            s3_subgraph_size=s3,
            s4_branching=s4,
            s5_dispersion=s5,
        ),
    )


def analyze_query(query: str, graph: nx.Graph) -> QueryAnalysisResult:
    features = compute_difficulty_features(query, graph)
    difficulty = compute_difficulty_score(features)
    return QueryAnalysisResult(
        query=query,
        features=features,
        difficulty=difficulty,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Analyze query against a Knowledge Graph"
    )
    parser.add_argument(
        "--graph", required=True, help="Path to the NetworkX JSON graph file"
    )
    parser.add_argument("--query", required=True, help="The query string to analyze")
    args = parser.parse_args()

    graph = load_graph(args.graph)
    result = analyze_query(args.query, graph)

    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
