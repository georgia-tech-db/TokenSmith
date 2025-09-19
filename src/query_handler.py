import networkx as nx
from src.llm_interface import get_json_response, get_text_response

def get_relevant_context_from_graph(query: str, graph: nx.Graph, model_name: str) -> str:
    """
    Retrieves context from the graph relevant to the user's query.
    """
    # Use LLM to extract key entities from the query
    prompt = f"""
    Extract the key entities from the following user query.
    Format the output as a JSON object with a single key "entities" which is a list of strings.
    Query: "{query}"
    """
    response = get_json_response(prompt, model_name)
    entities = response.get("entities", [])
    
    if not entities:
        return "I could not identify any key concepts in your question. Please be more specific."

    print(f"Identified entities in query: {entities}")
    
    context_parts = []
    found_nodes = set()
    relevant_communities = set()

    # Find nodes and communities related to the entities
    for entity in entities:
        if graph.has_node(entity):
            found_nodes.add(entity)
            community_id = graph.nodes[entity].get('community')
            if community_id is not None:
                relevant_communities.add(community_id)
            # Add neighbors for more context
            for neighbor in graph.neighbors(entity):
                found_nodes.add(neighbor)

    if not found_nodes:
        return "I could not find information related to your query in the textbook."
        
    community_summaries = graph.graph.get('community_summaries', {})
    for cid in relevant_communities:
        summary = community_summaries.get(str(cid)) # Ensure key is string
        if summary:
            context_parts.append(f"--- Relevant Topic Summary (Community {cid}) ---\n{summary}")

    if not context_parts:
        return "Found some concepts but could not retrieve a relevant topic summary."

    return "\n\n".join(context_parts)


def generate_graphrag_answer(query: str, context: str, model_name: str) -> str:
    """
    Generates a final, synthesized answer using the retrieved context.
    """
    prompt = f"""
    You are a helpful tutor. Based on the following context from a textbook, provide a clear and concise answer to the user's question.
    If the context does not contain the answer, state that the information is not available in the provided context.

    CONTEXT:
    {context}

    QUESTION: {query}

    ANSWER:
    """
    return get_text_response(prompt, model_name)