"""
This module supports several query transformation strategies using LLM calls.
"""
from typing import List
from src.generator import run_llama_cpp, text_cleaning

def hyde_transform(query: str, model_path: str) -> str:
    """
    Generates a hypothetical document for a given query.
    """
    print("Applying HyDE Transformation...")
    prompt = f"""<|im_start|>system
            You are a helpful assistant that generates a detailed, hypothetical answer to the user's question.
            <|im_end|>
            <|im_start|>user
            Please generate a hypothetical textbook paragraph that answers the following question.
            Question: {query}
            <|im_end|>
            <|im_start|>assistant
            """
    hypothetical_doc = run_llama_cpp(prompt, model_path, max_tokens=250)
    return text_cleaning(hypothetical_doc)

def multi_query_transform(query: str, model_path: str, num_queries: int = 3) -> List[str]:
    """
    Generates multiple variations of the original query.
    """
    print("Applying Multi-Query Transformation...")
    prompt = f"""<|im_start|>system
            You are a helpful assistant that generates {num_queries} different versions of a user's question to improve search retrieval. Each version should be on a new line.
            <|im_end|>
            <|im_start|>user
            Generate {num_queries} alternative questions for: {query}
            <|im_end|>
            <|im_start|>assistant
            """
    response = run_llama_cpp(prompt, model_path, max_tokens=200)
    queries = [q.strip() for q in response.strip().split('\n') if q.strip()]
    # Add the original query to the list
    queries.insert(0, query)
    return list(set(queries))

def decompose_query_transform(query: str, model_path: str) -> List[str]:
    """
    Decomposes a complex query into a list of sub-queries.
    """
    print("Applying Query Decomposition...")
    prompt = f"""<|im_start|>system
            You are a helpful assistant that breaks down a complex user question into a list of simpler, self-contained sub-questions. Each sub-question should be on a new line.
            <|im_end|>
            <|im_start|>user
            Decompose the following question into sub-questions: {query}
            <|im_end|>
            <|im_start|>assistant
            """
    response = run_llama_cpp(prompt, model_path, max_tokens=250)
    sub_queries = [q.strip() for q in response.strip().split('\n') if q.strip()]
    return sub_queries