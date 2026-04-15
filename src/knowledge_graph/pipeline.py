import logging
import os
import json
from time import time

import networkx as nx
import numpy as np

from src.knowledge_graph.models import Chunk, RunMetadata, CanonicalizationResult
from src.knowledge_graph.linkers import BaseLinker
from src.knowledge_graph.extractors import BaseExtractor
from src.knowledge_graph.canonicalizer import Canonicalizer


logger = logging.getLogger(__name__)


logger = logging.getLogger(__name__)


def build_kg(
    output_dir: str,
    chunks: list[Chunk],
    extractor: BaseExtractor,
    linker: BaseLinker,
    canonicalizer: Canonicalizer,
) -> nx.Graph:
    logger.info("Extracting keywords...")
    t0 = time()
    extractions = extractor.extract(chunks)
    t1 = time()
    logger.info(
        f"  {len(extractions)} extractions created in {t1 - t0:.2f} seconds"
    )

    logger.info("Canonicalizing keywords...")
    t0 = time()
    extractions, canon_result = canonicalizer.canonicalize(extractions)
    t1 = time()
    s = canon_result.stats
    logger.info(
        f"  {s['keywords_after_stage1']} → {s['canonical_keywords_final']} keywords, "
        f"{s['merges_performed']} merges, {s['llm_calls']} LLM calls "
        f"in {t1 - t0:.2f} seconds"
    )

    logger.info("Linking keywords...")
    t0 = time()
    graph = linker.link(extractions)
    t1 = time()
    logger.info(
        f"{graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges in {t1 - t0:.2f} seconds"
    )

    logger.info("Persisting graph...")
    t0 = time()
    run_metadata = RunMetadata(
        config={
            "extractor": extractor.get_config(),
            "linker": linker.get_config(),
        }, statistics={
            "extractor": extractor.metadata,
            "linker": linker.metadata,
        }
    )

    _persist(
        graph, chunks, output_dir,
        run_metadata=run_metadata,
    )
    t1 = time()
    logger.info(f"  Graph persisted in {t1 - t0:.2f} seconds")

    # Quick stats
    logger.info("═" * 50)
    logger.info(f"  Chunks:  {len(chunks)}")
    logger.info(f"  Nodes:   {graph.number_of_nodes()}")
    logger.info(f"  Edges:   {graph.number_of_edges()}")
    logger.info(f"  Output:  {output_dir}")
    logger.info("═" * 50)
    return graph


def _persist(
    graph: nx.Graph,
    chunks: list[Chunk],
    output_dir: str,
    run_metadata: RunMetadata | None = None,
    canonicalization_result: CanonicalizationResult | None = None,
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
        largest_comp_size = len(
            max(comp_list, key=len)) if comp_list else 0

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
            json.dump(run_metadata.to_dict(), f,
                      indent=2, ensure_ascii=False)

    if canonicalization_result is not None:
        with open(
            os.path.join(output_dir, "synonym_table.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(canonicalization_result.synonym_table, f, indent=2, ensure_ascii=False)

        with open(
            os.path.join(output_dir, "canonical_keywords.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(canonicalization_result.canonical_keywords, f, indent=2, ensure_ascii=False)

        np.save(
            os.path.join(output_dir, "canonical_embeddings.npy"),
            canonicalization_result.canonical_embeddings,
        )
