import json
import argparse
import networkx as nx
import matplotlib.pyplot as plt


def load_graph(filepath: str) -> nx.Graph:
    with open(filepath, "r") as f:
        data = json.load(f)
    return nx.node_link_graph(data)


def visualize_graph(graph: nx.Graph, output_file: str):
    plt.figure(figsize=(16, 12))
    pos = nx.spring_layout(graph, k=0.5, iterations=50)
    # Draw nodes
    nx.draw_networkx_nodes(graph, pos, node_size=300, node_color="lightblue", alpha=0.8)
    # Draw edges
    if graph.number_of_edges() > 0:
        weights = [graph[u][v].get("weight", 1) for u, v in graph.edges()]
        max_weight = max(weights) if weights else 1
        normalized_weights = [max(0.5, (w / max_weight) * 3) for w in weights]

        nx.draw_networkx_edges(
            graph, pos, width=normalized_weights, alpha=0.5, edge_color="gray"
        )
    # Draw labels
    nx.draw_networkx_labels(graph, pos, font_size=8, font_family="sans-serif")
    plt.title("Knowledge Graph Visualization")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Graph visualization saved to {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize a Knowledge Graph")
    parser.add_argument(
        "--graph", required=True, help="Path to the NetworkX JSON graph file"
    )
    parser.add_argument(
        "--output",
        default="graph_visualization.png",
        help="Output image file path (e.g., .png or .pdf)",
    )
    args = parser.parse_args()

    print(f"Loading graph from {args.graph}...")
    graph = load_graph(args.graph)
    print(
        f"Graph loaded. Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}"
    )

    print("Generating visualization...")
    visualize_graph(graph, args.output)


if __name__ == "__main__":
    main()
