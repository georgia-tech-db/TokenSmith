from itertools import combinations

import networkx as nx

from src.knowledge_graph.linkers import BaseLinker
from src.knowledge_graph.models import ExtractionResult


class CooccurrenceLinker(BaseLinker):
    """Create edges between every pair of nodes that co-occur in a chunk.

    For each :class:`ExtractionResult`, a pairwise edge is created (or
    updated) between every combination of nodes.  Edge ``weight`` is
    incremented on each co-occurrence and the ``chunk_ids`` list tracks
    which chunks contributed.

    Args:
        min_cooccurrence: Minimum edge weight to keep. Edges below this
            threshold are pruned after all chunks are processed.
    """

    def __init__(self, min_cooccurrence: int = 1):
        self.min_cooccurrence = min_cooccurrence

    def link(self, extractions: list[ExtractionResult]) -> nx.Graph:
        graph = nx.Graph()

        for extraction in extractions:
            chunk_id = extraction.chunk_id
            nodes = extraction.nodes

            # Ensure every node exists and track its chunk_ids
            for node in nodes:
                if graph.has_node(node):
                    graph.nodes[node]["chunk_ids"].append(chunk_id)
                else:
                    graph.add_node(node, chunk_ids=[chunk_id])

            # Create pairwise edges for co-occurrence
            for node_a, node_b in combinations(nodes, 2):
                if graph.has_edge(node_a, node_b):
                    graph[node_a][node_b]["weight"] += 1
                    graph[node_a][node_b]["chunk_ids"].append(chunk_id)
                else:
                    graph.add_edge(
                        node_a,
                        node_b,
                        weight=1,
                        chunk_ids=[chunk_id],
                    )

        # Prune edges below the minimum co-occurrence threshold
        if self.min_cooccurrence > 1:
            edges_to_remove = [
                (u, v)
                for u, v, data in graph.edges(data=True)
                if data["weight"] < self.min_cooccurrence
            ]
            graph.remove_edges_from(edges_to_remove)

            # Remove isolated nodes left after pruning
            isolates = list(nx.isolates(graph))
            graph.remove_nodes_from(isolates)

        return graph
