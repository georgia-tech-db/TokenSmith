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
        You are a helpful assistant. Generate a brief, factual answer to the following question.
        Write as if you are a textbook excerpt. Be specific and use technical terms.
        The book excerpt should be as if they are from: Database System Concepts Seventh Edition by Avi Silberschatz, Henry F. Korth, S. Sudarshan.
        <|im_end|>
        <|im_start|>user
        Question: {query}
        
        Write a brief answer (2-3 sentences). End the answer with {ANSWER_END}.
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


def enrich_query_with_llm(
    query: str,
    model_path: str,
    max_tokens: int = 60,
    **llm_kwargs
) -> str:
    """
    NOTE: This should not be used together with HyDE, as they serve similar purposes.
    LLM-based query enrichment (not used by default).
    Expands abbreviations, adds synonyms and related terms.    
    Example: "What is PK?" -> "What is a primary key (PK)? Unique identifier constraint."
    """
    prompt = textwrap.dedent(f"""\
        <|im_start|>system
        You are a query expansion expert. Expand the query by adding relevant keywords,
        synonyms, and related terms. Keep it concise (1-2 sentences).
        <|im_end|>
        <|im_start|>user
        Original query: {query}
        
        Enriched query:
        <|im_end|>
        <|im_start|>assistant
        """)
    
    prompt = text_cleaning(prompt)
    enriched = run_llama_cpp(
        prompt,
        model_path,
        max_tokens=max_tokens,
        temperature=0.4,
        **llm_kwargs
    )
    
    return enriched.strip()
