import argparse
import yaml
import os
import networkx as nx
import json 

from src.preprocess import build_index
from src.retriever import retrieve, load_artifacts
from src.ranker import rerank
from src.generator import answer
from src.graph_builder import build_graph_index
from src.query_handler import get_relevant_context_from_graph, generate_graphrag_answer

def main():
    parser = argparse.ArgumentParser(description="Textbook Q&A with RAG and GraphRAG")
    subparsers = parser.add_subparsers(dest="mode", required=True)
    index_parser = subparsers.add_parser("index", help="Build a FAISS vector index.")
    chat_parser = subparsers.add_parser("chat", help="Chat using the FAISS vector index.")
    graphrag_parser = subparsers.add_parser("graphrag", help="Use GraphRAG functionality.")
    graphrag_subparsers = graphrag_parser.add_subparsers(dest="subcommand", required=True)
    graphrag_build = graphrag_subparsers.add_parser("build", help="Build the knowledge graph.")
    graphrag_chat = graphrag_subparsers.add_parser("chat", help="Chat using the knowledge graph.")
    for p in [index_parser, chat_parser, graphrag_build, graphrag_chat]:
        p.add_argument("--config", default="config/config.yaml")
        p.add_argument("--pdf_dir", default="data/chapters/")
        p.add_argument("--index_prefix", default="output/textbook_index")
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    if args.mode == "graphrag":
        graph_output_path = os.path.join(cfg['graph_output_dir'], cfg['graph_file_name'])
        
        if args.subcommand == "build":
            build_graph_index(cfg, args.pdf_dir)

        elif args.subcommand == "chat":
            if not os.path.exists(graph_output_path):
                print(f"Graph file not found at {graph_output_path}. Please run 'graphrag build' first.")
                return

            print("Loading knowledge graph...")
            graph = nx.read_graphml(graph_output_path)
            print("Graph loaded successfully")
            
            if 'community_summaries' in graph.graph and isinstance(graph.graph['community_summaries'], str):
                try:
                    graph.graph['community_summaries'] = json.loads(graph.graph['community_summaries'])
                except json.JSONDecodeError:
                    print("Warning: Could not parse community_summaries from graph file.")
                    graph.graph['community_summaries'] = {}

            print("📚 Ready with GraphRAG. Type 'exit' to quit.")
            while True:
                query = input("\nAsk (GraphRAG) > ").strip()
                if query.lower() == 'exit': break

                context = get_relevant_context_from_graph(query, graph, cfg['llm_model'])
                final_answer = generate_graphrag_answer(query, context, cfg['llm_model'])

                print("\n=== ANSWER =========================================")
                print(final_answer.strip() or "(No output from model)")
                print("====================================================\n")

if __name__ == "__main__":
    main()