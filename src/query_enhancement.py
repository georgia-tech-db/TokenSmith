"""
Query enhancement techniques for improved retrieval (use only one):
- HyDE (Hypothetical Document Embeddings): Generate hypothetical answer for better retrieval
- Query Enrichment: LLM-based query expansion
"""

import textwrap
from typing import Optional
from src.generator import ANSWER_END, ANSWER_START, run_llama_cpp, text_cleaning


def generate_hypothetical_document(
    query: str,
    model_path: str,
    max_tokens: int = 100,
    **llm_kwargs
) -> str:
    """
    HyDE: Generate a hypothetical answer to improve retrieval quality.
    Concept: Hypothetical answers are semantically closer to actual documents than queries.
    Ref: https://arxiv.org/abs/2212.10496
    """
    prompt = textwrap.dedent(f"""\
        <|im_start|>system
        You are a database systems expert. Generate a concise, technical answer using precise database terminology.
        Write in the formal academic style of Database System Concepts (Silberschatz, Korth, Sudarshan).
        Use specific terms for: relational model concepts (relations, tuples, attributes, keys, schemas), 
        SQL and query languages, transactions (ACID properties, concurrency control, recovery), 
        storage structures (indexes, B+ trees), normalization (functional dependencies, normal forms), 
        and database design (E-R model, decomposition).
        Focus on definitions, mechanisms, and technical accuracy rather than examples.
        <|im_end|>
        <|im_start|>user
        Question: {query}
        
        Generate a precise answer (2-4 sentences) using appropriate technical terminology. End with {ANSWER_END}.
        <|im_end|>
        <|im_start|>assistant
        {ANSWER_START}
        """)
    
    prompt = text_cleaning(prompt)
    hypothetical = run_llama_cpp(
        prompt,
        model_path,
        max_tokens=max_tokens,
        **llm_kwargs
    )
    
    return hypothetical.strip()

def generate_multi_queries(
    query: str,
    model_path: str,
    n: int = 3,
    **llm_kwargs
) -> list[str]:
    """
    Generate multiple search query variations to improve recall.
    """
    prompt = textwrap.dedent(f"""\
        <|im_start|>system
        You are a helpful assistant that generates search queries.
        <|im_end|>
        <|im_start|>user
        Generate {n} different search queries related to the following user question.
        The queries should cover different aspects or terminology of the question to maximize retrieval coverage.
        Return ONLY the queries, one per line. Do not number them.
        
        User Question: {query}
        <|im_end|>
        <|im_start|>assistant
        {ANSWER_START}
        """)
    
    prompt = text_cleaning(prompt)
    raw_output = run_llama_cpp(
        prompt,
        model_path,
        max_tokens=100,
        **llm_kwargs
    )
    
    # Parse output: split by newlines and clean up
    queries = [q.strip() for q in raw_output.split('\n') if q.strip()]
    return queries[:n]
