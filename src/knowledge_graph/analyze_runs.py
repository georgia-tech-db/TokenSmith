import os
import json
import glob
import argparse
import networkx as nx


def format_extractor_config(extractor_cfg):
    """Format extractor configuration for display."""
    cls_name = extractor_cfg.get("class", "Unknown")
    if cls_name == "CompositeExtractor":
        sub_extractors = [
            e.get("class", "Unknown") for e in extractor_cfg.get("extractors", [])
        ]
        return f"Composite({', '.join(sub_extractors)})"
    return cls_name


def analyze_runs(output_dir):
    """Scan and analyze metadata files in the output directory."""
    pattern = os.path.join(output_dir, "run_metadata__*.json")
    files = glob.glob(pattern)

    if not files:
        print(f"No metadata files found in {output_dir}")
        return

    runs = []
    for file_path in files:
        try:
            # Extract timestamp from filename
            filename = os.path.basename(file_path)
            timestamp_str = filename.replace("run_metadata__", "").replace(".json", "")

            with open(file_path, "r") as f:
                data = json.load(f)

            config = data.get("config", {})
            stats = data.get("statistics", {})

            graph_stats = stats.get("graph", {})
            run_info = {
                "timestamp": timestamp_str,
                "extractor": format_extractor_config(config.get("extractor", {})),
                "min_cooc": config.get("linker", {}).get("min_cooccurrence", "N/A"),
                "deleted_edges": stats.get("linker", {}).get("deleted_edges", 0),
                "deleted_nodes": stats.get("linker", {}).get("deleted_nodes", 0),
                "final_nodes": graph_stats.get("nodes", 0),
                "final_edges": graph_stats.get("edges", 0),
                "file": filename,
            }
            runs.append(run_info)
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")

    # Sort by timestamp
    runs.sort(key=lambda x: x["timestamp"])

    # Header
    print("\n" + "=" * 100)
    print(
        f"{'Timestamp':<20} | {'Extractor':<25} | {'MinCooc':<7} | {'DelEdges':<8} | {'DelNodes':<8} | {'Nodes':<5} | {'Edges':<5}"
    )
    print("-" * 110)

    for run in runs:
        print(
            f"{run['timestamp']:<20} | {run['extractor']:<25} | {run['min_cooc']:<7} | {run['deleted_edges']:<8} | {run['deleted_nodes']:<8} | {run['final_nodes']:<5} | {run['final_edges']:<5}"
        )

    print("=" * 100 + "\n")

    # Basic analysis
    if runs:
        min_edges_run = min(runs, key=lambda x: x["deleted_edges"])
        max_edges_run = max(runs, key=lambda x: x["deleted_edges"])

        print("Analysis Summary:")
        print(
            f"  Run with LEAST deleted edges: {min_edges_run['timestamp']} ({min_edges_run['deleted_edges']} edges, {min_edges_run['extractor']})"
        )
        print(
            f"  Run with MOST deleted edges:  {max_edges_run['timestamp']} ({max_edges_run['deleted_edges']} edges, {max_edges_run['extractor']})"
        )

        min_nodes_run = min(runs, key=lambda x: x["deleted_nodes"])
        print(
            f"  Run with LEAST deleted nodes: {min_nodes_run['timestamp']} ({min_nodes_run['deleted_nodes']} nodes, {min_nodes_run['extractor']})"
        )

    # Graph Comparison (if there are at least 2 runs)
    if len(runs) >= 2:
        print("\n" + "=" * 110)
        print(f"{'Graph Comparison (Pairwise)':^110}")
        print("-" * 110)

        # We'll compare each run with the NEXT one (chronologically)
        for i in range(len(runs) - 1):
            run1 = runs[i]
            run2 = runs[i + 1]

            graph1_path = os.path.join(output_dir, f"graph__{run1['timestamp']}.json")
            graph2_path = os.path.join(output_dir, f"graph__{run2['timestamp']}.json")

            if not (os.path.exists(graph1_path) and os.path.exists(graph2_path)):
                print(
                    f"Skipping comparison: Missing graph files for {run1['timestamp']} or {run2['timestamp']}"
                )
                continue

            try:
                with open(graph1_path, "r") as f:
                    g1 = nx.node_link_graph(json.load(f))
                with open(graph2_path, "r") as f:
                    g2 = nx.node_link_graph(json.load(f))

                nodes1 = set(g1.nodes())
                nodes2 = set(g2.nodes())
                edges1 = set(tuple(sorted((u, v))) for u, v in g1.edges())
                edges2 = set(tuple(sorted((u, v))) for u, v in g2.edges())

                common_nodes = nodes1.intersection(nodes2)
                union_nodes = nodes1.union(nodes2)
                jaccard_nodes = (
                    len(common_nodes) / len(union_nodes) if union_nodes else 0
                )

                common_edges = edges1.intersection(edges2)
                union_edges = edges1.union(edges2)
                jaccard_edges = (
                    len(common_edges) / len(union_edges) if union_edges else 0
                )

                diff1 = nodes1 - nodes2
                diff2 = nodes2 - nodes1

                print(f"Comparison: {run1['timestamp']} -> {run2['timestamp']}")
                print(
                    f"  Node Jaccard:      {jaccard_nodes:.4f} ({len(common_nodes)} common nodes out of {len(union_nodes)})"
                )
                print(
                    f"  Edge Jaccard:      {jaccard_edges:.4f} ({len(common_edges)} common edges out of {len(union_edges)})"
                )
                print(f"  Unique to '{run1['extractor']}': {len(diff1)} nodes")
                print(f"  Unique to '{run2['extractor']}': {len(diff2)} nodes")

                if diff2:
                    sorted_diff2 = sorted(list(diff2))
                    print(
                        f"  Example new nodes: {', '.join(sorted_diff2[:8])}{'...' if len(sorted_diff2) > 8 else ''}"
                    )

                # Compare average degree shift
                avg_deg1 = sum(dict(g1.degree()).values()) / len(g1) if len(g1) else 0
                avg_deg2 = sum(dict(g2.degree()).values()) / len(g2) if len(g2) else 0
                print(f"  Avg Degree Shift:  {avg_deg1:.2f} -> {avg_deg2:.2f}")

                # Top nodes overlap
                top10_1 = sorted(g1.degree(), key=lambda x: x[1], reverse=True)[:10]
                top10_2 = sorted(g2.degree(), key=lambda x: x[1], reverse=True)[:10]

                top10_nodes1 = set(n for n, d in top10_1)
                top10_nodes2 = set(n for n, d in top10_2)
                common_top = top10_nodes1.intersection(top10_nodes2)

                print(f"  Top 10 Nodes Overlap: {len(common_top)}/10 common")
                if common_top:
                    print(f"    Shared central nodes: {', '.join(list(common_top))}")

                print("-" * 50)

            except Exception as e:
                print(f"Error during graph comparison: {e}")

        print("=" * 110 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze knowledge graph pipeline runs."
    )
    parser.add_argument(
        "--dir",
        default="data/knowledge_graph",
        help="Directory containing metadata files.",
    )
    args = parser.parse_args()

    # Try to find the directory relative to project root if it doesn't exist directly
    dir_to_scan = args.dir
    if not os.path.exists(dir_to_scan):
        # Assume we might be in src/knowledge_graph
        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        alt_dir = os.path.join(project_root, "data", "knowledge_graph")
        if os.path.exists(alt_dir):
            dir_to_scan = alt_dir

    analyze_runs(dir_to_scan)
