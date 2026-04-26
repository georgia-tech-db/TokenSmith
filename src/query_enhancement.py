"""Query enhancement helpers for HyDE, grammar correction, decomposition, and follow-up rewriting."""

import re
import textwrap

from src.generator import ANSWER_END, ANSWER_START, run_llama_cpp, text_cleaning
from src.planning.rules import (
    FINAL_FOLLOW_UP_REFERENCES,
    FOLLOW_UP_REFERENCE_PATTERN,
    ORDINAL_FOLLOW_UP_REFERENCES,
)


class QueryEnhancementError(RuntimeError):
    """Raised when an LLM-backed query enhancement step cannot produce valid output."""


def _completion_text(output) -> str:
    """Extract text from llama.cpp completion output, tolerating string test doubles."""
    if isinstance(output, str):
        return output
    try:
        return output["choices"][0]["text"]
    except (KeyError, IndexError, TypeError):
        raise QueryEnhancementError("LLM completion output did not contain choices[0].text") from None


QUESTION_PREFIX_PATTERN = re.compile(
    r"^(?:what\s+(?:is|are)|define|definition\s+of|explain|how\s+(?:does|do|is|are)|why\s+(?:does|do|is|are)|compare|contrast)\s+",
    re.IGNORECASE,
)
STOP_PHRASES_PATTERN = re.compile(r"\b(?:in|for|of|with|about|from)\s+(?:database|databases|dbms|systems?)\b", re.IGNORECASE)
ENUMERATION_PATTERN = re.compile(r"\b([A-Z][A-Za-z0-9+#-]*(?:\s+[A-Z][A-Za-z0-9+#-]*)?)\b")
ENUMERATION_INTRO_PATTERN = re.compile(
    r"\b(?:stands\s+for|are|include|includes|consist\s+of|consists\s+of)\b",
    re.IGNORECASE,
)
def _clean_topic(text: str) -> str:
    """Extract a compact technical topic phrase from a previous user question."""
    topic = text.strip().strip("?.! ")
    topic = QUESTION_PREFIX_PATTERN.sub("", topic)
    topic = STOP_PHRASES_PATTERN.sub("", topic)
    topic = re.sub(r"\s+", " ", topic).strip(" ,;:")
    topic = re.sub(r"^(?:a|an|the)\s+", "", topic, flags=re.IGNORECASE)
    return topic


def _latest_user_topic(history: list[dict]) -> str | None:
    """Return the most recent non-empty technical topic mentioned by the user."""
    for turn in reversed(history):
        if turn.get("role") != "user":
            continue
        topic = _clean_topic(str(turn.get("content", "")))
        if topic:
            return topic
    return None


def _clean_enumerated_term(term: str) -> str:
    """Normalize one candidate term from an enumerated assistant answer."""
    cleaned = re.sub(r"\s+", " ", term).strip(" ,;:.!?")
    cleaned = re.sub(r"^(?:and|or|the|a|an)\s+", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip(" ,;:.!?")


def _split_enumerated_terms(text: str) -> list[str]:
    """Extract comma/and-delimited technical terms from an explanatory answer."""
    intro_match = ENUMERATION_INTRO_PATTERN.search(text)
    if not intro_match:
        return []

    tail = text[intro_match.end():]
    tail = re.split(r"[.\n]", tail, maxsplit=1)[0]
    tail = re.sub(r"\s+(?:and|or)\s+", ", ", tail, flags=re.IGNORECASE)

    terms: list[str] = []
    for fragment in tail.split(","):
        term = _clean_enumerated_term(fragment)
        if not term or len(term.split()) > 5:
            continue
        if term.lower() in {existing.lower() for existing in terms}:
            continue
        terms.append(term)
    return terms


def _ordered_answer_terms(history: list[dict]) -> list[str]:
    """Extract ordered terms from the last assistant answer for ordinal follow-ups."""
    for turn in reversed(history):
        if turn.get("role") != "assistant":
            continue
        text = str(turn.get("content", ""))
        deduped = _split_enumerated_terms(text)
        if not deduped:
            for match in ENUMERATION_PATTERN.finditer(text):
                term = _clean_enumerated_term(match.group(1))
                if term and term.lower() not in {existing.lower() for existing in deduped}:
                    deduped.append(term)
        if deduped:
            return deduped
    return []


def deterministic_contextualize_query(query: str, history: list[dict]) -> str | None:
    """Rewrite simple follow-ups with explicit entities from recent history."""
    if not history or not FOLLOW_UP_REFERENCE_PATTERN.search(query):
        return None

    rewritten = query.strip()
    ordered_terms = _ordered_answer_terms(history)
    for phrase, index in ORDINAL_FOLLOW_UP_REFERENCES:
        if index < len(ordered_terms):
            rewritten = re.sub(rf"\b{re.escape(phrase)}\b", ordered_terms[index], rewritten, flags=re.IGNORECASE)
    if ordered_terms:
        for phrase in FINAL_FOLLOW_UP_REFERENCES:
            rewritten = re.sub(rf"\b{re.escape(phrase)}\b", ordered_terms[-1], rewritten, flags=re.IGNORECASE)

    topic = _latest_user_topic(history)
    if topic:
        rewritten = re.sub(
            r"\bsuch\s+(?:a|an|the)?\s*[a-z][a-z0-9-]*\b",
            topic,
            rewritten,
            flags=re.IGNORECASE,
        )
        rewritten = re.sub(
            r"\b(it|that|this|they|them|those|these)\b",
            topic,
            rewritten,
            flags=re.IGNORECASE,
        )
        rewritten = re.sub(r"\b(its|their)\b", f"{topic}'s", rewritten, flags=re.IGNORECASE)

    return rewritten if rewritten != query.strip() else None


def _require_non_empty(text: str, step: str) -> str:
    """Return stripped text or raise when an LLM helper produced no content."""
    cleaned = text.strip()
    if not cleaned:
        raise QueryEnhancementError(f"{step} produced an empty response")
    return cleaned


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
        
        Generate a precise and a concise answer (2-4 sentences) using appropriate technical terminology. End with {ANSWER_END}.
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
    return _require_non_empty(_completion_text(hypothetical), "HyDE generation")

def correct_query_grammar(
    query: str,
    model_path: str,
    **llm_kwargs
) -> str:
    """
    Corrects spelling and grammatical errors in the query to improve keyword matching.
    """
    prompt = textwrap.dedent(f"""\
        <|im_start|>system
        You are a helpful assistant that corrects search queries.
        Your task is to correct any spelling or grammatical errors in the user's query.
        Do not answer the question. Output ONLY the corrected query.
        <|im_end|>
        <|im_start|>user
        Original Query: {query}
        <|im_end|>
        <|im_start|>assistant
        """)

    prompt = text_cleaning(prompt)
    corrected_query = run_llama_cpp(
        prompt,
        model_path,
        max_tokens=len(query.split()) * 2,
        temperature=0,
        **llm_kwargs
    )

    # If model returns empty or hallucinated long text, return original
    cleaned = _completion_text(corrected_query).strip()
    if not cleaned or len(cleaned) > len(query) * 2:
        raise QueryEnhancementError("Grammar correction produced an invalid query")

    return cleaned

def expand_query_with_keywords(
    query: str,
    model_path: str,
    max_tokens: int = 64,
    **llm_kwargs
) -> list[str]:
    """
    Query Expansion: Generates related keywords and synonyms.
    This helps retrieval when the user uses different vocabulary than the documents.
    """
    prompt = textwrap.dedent(f"""\
        <|im_start|>system
        You are a search optimization expert.
        Generate 3 alternative versions of the user's query using synonyms and related technical terms.
        Output the alternative queries separated by newlines. Do not provide explanations.
        <|im_end|>
        <|im_start|>user
        Query: {query}
        <|im_end|>
        <|im_start|>assistant
        """)

    prompt = text_cleaning(prompt)
    expansion = run_llama_cpp(
        prompt,
        model_path,
        max_tokens=max_tokens,
        temperature=0.5,
        **llm_kwargs
    )

    # Combine original query with expansion
    query_lines = [query]
    query_lines.extend([line.strip() for line in _completion_text(expansion).split('\n') if line.strip()])

    # Remove numbering if present
    query_lines = [line.split('.', 1)[-1].strip() if '.' in line[:3] else line for line in query_lines]

    if len(query_lines) == 1:
        raise QueryEnhancementError("Keyword expansion produced no alternatives")
    return query_lines


def decompose_complex_query(
    query: str,
    model_path: str,
    **llm_kwargs
) -> list[str]:
    """
    Breaks a complex multi-part question into sub-questions.
    Useful for tasks where a single retrieval might miss some parts of the answer.
    """
    prompt = textwrap.dedent(f"""\
        <|im_start|>system
        Break the following complex question into simple, single-step sub-questions.
        If the question is already simple, just output the original question.
        Output each sub-question on a new line. Do not provide explanations.
        <|im_end|>
        <|im_start|>user
        Complex Question: {query}
        <|im_end|>
        <|im_start|>assistant
        """)

    prompt = text_cleaning(prompt)
    output = run_llama_cpp(
        prompt,
        model_path,
        max_tokens=128,
        temperature=0.0,
        **llm_kwargs
    )

    sub_questions = [line.strip() for line in _completion_text(output).split('\n') if line.strip()]

    # Remove numbering if present
    sub_questions = [line.split('.', 1)[-1].strip() if '.' in line[:3] else line for line in sub_questions]

    if not sub_questions:
        raise QueryEnhancementError("Query decomposition produced no sub-questions")
    return sub_questions

def contextualize_query(
    query: str,
    history: list[dict],
    model_path: str,
    max_tokens: int = 128,
    **llm_kwargs
) -> str:
    """
    Rewrites a query to be standalone based on chat history.
    """
    if not history:
        return query

    deterministic_rewrite = deterministic_contextualize_query(query, history)
    if deterministic_rewrite:
        return deterministic_rewrite

    # Format history into a compact string
    # We expect history to be list of dicts: [{"role": "user", "content": "..."}, ...]
    conversation_text = ""
    for turn in history[-4:]:  # Only look at the last two user/assistant turns.
        role = "User" if turn["role"] == "user" else "Assistant"
        content = turn["content"]
        conversation_text += f"{role}: {content}\n"

    prompt = textwrap.dedent(f"""\
        <|im_start|>system
        You are a query rewriting assistant. Your task is to rewrite the user's "Follow Up Input" to be a standalone question by replacing pronouns (it, they, this, that) with specific nouns from the "Chat History".
        
        Examples:
        History:
        User: What is BCNF?
        Assistant: It is a normal form used in database normalization.
        Input: Why is it useful?
        Output: Why is BCNF useful?
        
        History:
        User: Explain the ACID properties.
        Assistant: ACID stands for Atomicity, Consistency, Isolation, Durability.
        Input: Give me an example of the first one.
        Output: Give me an example of Atomicity.

        History:
        User: Who created Python?
        Assistant: Guido van Rossum.
        Input: what is sql?
        Output: what is sql?
        <|im_end|>
        <|im_start|>user
        Chat History:
        {conversation_text}
        
        Follow Up Input: {query}
        
        Output:
        <|im_end|>
        <|im_start|>assistant
        """)

    prompt = text_cleaning(prompt)
    output = run_llama_cpp(
        prompt,
        model_path,
        max_tokens=max_tokens,
        temperature=0.0,
        **llm_kwargs
    )

    rewritten = _completion_text(output).strip()
    
    # If model hallucinates or errors, fall back to original query
    if not rewritten or len(rewritten) > len(query) * 2:
        raise QueryEnhancementError("Follow-up rewriting produced an invalid query")
        
    return rewritten
