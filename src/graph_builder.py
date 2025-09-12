import os
import networkx as nx
import igraph as ig
import leidenalg as la
from tqdm import tqdm
import fitz # PyMuPDF
import json

from src.llm_interface import extract_graph_elements, summarize_text
from src.chunking import make_chunk_strategy
from src.preprocess import _resolve_pdf_paths

def build_knowledge_graph(text_chunks: list, model_name: str):
    graph = nx.Graph()
    print("Extracting entities and relationships from chunks...")
    for chunk_info in tqdm(text_chunks, desc="Processing Chunks"):
        response = extract_graph_elements(chunk_info['text'], model_name)
        
        if response and "elements" in response and isinstance(response["elements"], list):
            for el in response["elements"]:
                if isinstance(el, list) and len(el) == 3 and all(isinstance(i, str) for i in el):
                    source_entity, relationship, target_entity = el
                    if not source_entity or not target_entity: continue
                    source_chunks = graph.nodes.get(source_entity, {}).get('chunk_ids', set())
                    target_chunks = graph.nodes.get(target_entity, {}).get('chunk_ids', set())
                    graph.add_node(source_entity, chunk_ids=source_chunks.union({chunk_info['chunk_id']}))
                    graph.add_node(target_entity, chunk_ids=target_chunks.union({chunk_info['chunk_id']}))
                    graph.add_edge(source_entity, target_entity, label=relationship)
                else:
                    print(f"\n[Warning] Malformed element skipped: {el}")
    return graph

def add_community_detection(graph):
    print("Detecting communities...")
    if graph.number_of_nodes() == 0:
        print("Graph is empty, skipping community detection.")
        return graph, None
    igraph_graph = ig.Graph.from_networkx(graph)
    partition = la.find_partition(igraph_graph, la.ModularityVertexPartition)
    for vertex in igraph_graph.vs:
        node_name = vertex['_nx_name']
        community_id = partition.membership[vertex.index]
        graph.nodes[node_name]['community'] = community_id
    print(f"Detected {len(partition)} communities.")
    return graph, partition

def add_summaries_to_graph(graph, partition, text_chunks: list, model_name: str):
    if partition is None:
        print("No partition found, skipping summarization.")
        return graph
    print("Summarizing communities...")
    chunk_map = {chunk['chunk_id']: chunk['text'] for chunk in text_chunks}
    community_docs = {}
    for node, data in graph.nodes(data=True):
        community_id = data.get('community')
        chunk_ids = data.get('chunk_ids', set())
        if community_id is not None:
            if community_id not in community_docs: community_docs[community_id] = set()
            for cid in chunk_ids:
                if cid in chunk_map: community_docs[community_id].add(chunk_map[cid])
    summaries = {}
    for cid, docs in tqdm(community_docs.items(), desc="Summarizing"):
        full_text = " ".join(list(docs))
        if full_text.strip():
            summary = summarize_text(full_text, model_name)
            summaries[cid] = summary
    graph.graph['community_summaries'] = summaries
    return graph

def build_graph_index(config: dict, pdf_dir: str):
    output_path = os.path.join(config['graph_output_dir'], config['graph_file_name'])
    pdf_paths = _resolve_pdf_paths(pdf_dir, None, None)
    if not pdf_paths: raise FileNotFoundError(f"No PDFs found in {pdf_dir}")

    print("Step 1: Loading and Chunking Text from PDFs...")
    all_chunks = []
    chunk_strategy = make_chunk_strategy("sliding-tokens", chunk_size_char=0, chunk_tokens=config['graph_chunk_size'], tokenizer_name=config['embed_model'])
    if hasattr(chunk_strategy, 'overlap_tokens'):
        chunk_strategy.overlap_tokens = config['graph_overlap']
    for path in tqdm(pdf_paths, desc="Extracting Text"):
        with fitz.open(path) as doc:
            full_text = "".join(page.get_text() for page in doc).replace('\n', ' ').strip()
            if full_text:
                chunks = chunk_strategy.chunk(full_text)
                for i, chunk_text in enumerate(chunks):
                    all_chunks.append({"chunk_id": f"{os.path.basename(path)}-{i}", "text": chunk_text})
    print(f"\nCreated {len(all_chunks)} text chunks.")
    
    print("\nStep 2: Building Initial Knowledge Graph...")
    graph = build_knowledge_graph(all_chunks, config['llm_model'])
    print(f"\nInitial graph has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
    
    print("\nStep 3: Detecting Communities...")
    graph, partition = add_community_detection(graph)
    
    print("\nStep 4: Summarizing Communities...")
    final_graph = add_summaries_to_graph(graph, partition, all_chunks, config['llm_model'])
    
    print(f"\nSaving graph to {output_path}")
    
    # Convert all complex attributes to strings before saving.
    for node, data in final_graph.nodes(data=True):
        # Convert the chunk_ids set to a comma-separated string
        if 'chunk_ids' in data and isinstance(data['chunk_ids'], set):
            data['chunk_ids'] = ",".join(map(str, data['chunk_ids']))

    if 'community_summaries' in final_graph.graph and isinstance(final_graph.graph['community_summaries'], dict):
        final_graph.graph['community_summaries'] = json.dumps(final_graph.graph['community_summaries'])

    nx.write_graphml(final_graph, output_path)
    print("Graph building complete.")