import json
import argparse
import os
import logging

from src.knowledge_graph.analysis import analyze_query
from src.knowledge_graph.io import RUNS_DIR, load_graph
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze query difficulty against a Knowledge Graph."
    )
    parser.add_argument(
        "--graph",
        default=os.path.join(RUNS_DIR, "latest", "graph.json"),
        help="Path to the NetworkX JSON graph file (default: latest run).",
    )
    parser.add_argument("--query", required=True, help="The query string to analyze.")
    parser.add_argument("--debug", action="store_true", help="Print debug information during analysis.")
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(name)s  %(levelname)s %(message)s")

    graph = load_graph(args.graph)
    logger.debug(f"Loaded graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
    result = analyze_query(args.query, graph)
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
