import json
import os
from time import time
import networkx as nx

from src.knowledge_graph.persisters import BasePersister
from src.knowledge_graph.models import Chunk


class NetworkxJsonPersister(BasePersister):
    """Save the graph in NetworkX node-link JSON format and chunks as a
    separate JSON dictionary.

    Output files:

    * ``graph.json`` — NetworkX node-link serialization (nodes, edges,
      attributes).
    * ``chunks.json`` — ``{ "0": "chunk text …", "1": "…" }`` indexed
      by chunk ID.
    """

    def persist(self, graph: nx.Graph, chunks: list[Chunk], output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)

        # --- graph.json ---
        graph_data = nx.node_link_data(graph)
        graph_path = os.path.join(output_dir, "graph_{}.json".format(time()))
        with open(graph_path, "w", encoding="utf-8") as f:
            json.dump(graph_data, f, indent=2, ensure_ascii=False)

        # --- chunks.json ---
        chunk_store = {str(chunk.id): chunk.text for chunk in chunks}
        chunks_path = os.path.join(output_dir, "chunks.json")
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(chunk_store, f, indent=2, ensure_ascii=False)
