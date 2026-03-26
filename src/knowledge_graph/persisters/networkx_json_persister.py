import json
import os

import networkx as nx

from src.knowledge_graph.persisters import BasePersister
from src.knowledge_graph.models import Chunk, RunMetadata



class NetworkxJsonPersister(BasePersister):
    """Save the graph in NetworkX node-link JSON format and chunks as a
    separate JSON dictionary.

    The caller is responsible for creating a timestamped run directory and
    passing it as ``output_dir``.  This persister writes fixed filenames into
    that directory so the directory itself carries the run identity:

    * ``graph.json``               — NetworkX node-link serialization
    * ``chunks.json``              — ``{ "0": "chunk text …", "1": "…" }``
    * ``run_metadata.json``        — timing + graph statistics (optional)
    """

    def persist(
        self,
        graph: nx.Graph,
        chunks: list[Chunk],
        output_dir: str,
        run_metadata: RunMetadata | None = None,
    ) -> None:
        os.makedirs(output_dir, exist_ok=True)

        graph_data = nx.node_link_data(graph)
        with open(os.path.join(output_dir, "graph.json"), "w", encoding="utf-8") as f:
            json.dump(graph_data, f, indent=2, ensure_ascii=False)

        chunk_store = {str(chunk.id): chunk.text for chunk in chunks}
        with open(os.path.join(output_dir, "chunks.json"), "w", encoding="utf-8") as f:
            json.dump(chunk_store, f, indent=2, ensure_ascii=False)

        if run_metadata:
            num_nodes = graph.number_of_nodes()
            num_edges = graph.number_of_edges()
            comp_list = list(nx.connected_components(graph))
            largest_comp_size = len(max(comp_list, key=len)) if comp_list else 0

            run_metadata.statistics["graph"] = {
                "nodes": num_nodes,
                "edges": num_edges,
                "density": nx.density(graph),
                "avg_degree": (2 * num_edges / num_nodes) if num_nodes > 0 else 0.0,
                "avg_clustering": nx.average_clustering(graph),
                "num_connected_components": len(comp_list),
                "largest_component_size": largest_comp_size,
                "max_degree": max(dict(graph.degree()).values(), default=0),
            }
            with open(
                os.path.join(output_dir, "run_metadata.json"), "w", encoding="utf-8"
            ) as f:
                json.dump(run_metadata.to_dict(), f, indent=2, ensure_ascii=False)
