import textwrap
import re
from src.llm_interface import get_text_response # <-- NEW IMPORT

ANSWER_START = "<<<ANSWER>>>"
ANSWER_END   = "<<<END>>>"

def text_cleaning(prompt):
    _CONTROL_CHARS_RE = re.compile(r'[\u0000-\u001F\u007F-\u009F]')
    _DANGEROUS_PATTERNS = [
        r'ignore\s+(all\s+)?previous\s+instructions?',
        r'you\s+are\s+now\s+(in\s+)?developer\s+mode',
    ]
    text = _CONTROL_CHARS_RE.sub('', prompt)
    text = re.sub(r'\s+', ' ', text).strip()
    for pat in _DANGEROUS_PATTERNS:
        text = re.sub(pat, '[FILTERED]', text, flags=re.IGNORECASE)
    return text

def format_prompt(chunks, query, max_chunk_chars=400):
    trimmed = [(c or "")[:max_chunk_chars] for c in chunks]
    context = "\n\n".join(trimmed)
    context = text_cleaning(context)
    return textwrap.dedent(f"""\
        <|im_start|>system
        You are a helpful tutor. Use the attached document excerpts to answer the user's query concisely.
        If the excerpts do not contain the answer, say so.
        <|im_end|>
        <|im_start|>user
        Textbook Excerpts:
        {context}

        Question: {query}
        <|im_end|>
        <|im_start|>assistant
    """)

def answer(query: str, chunks, model_name: str, **kw):
    """
    Generates an answer using the classic RAG approach with Ollama.
    """
    prompt = format_prompt(chunks, query)
    print(f"\n⚙️  Generating answer with classic RAG prompt...")
    
    # Use the new Ollama interface
    response = get_text_response(prompt, model_name)
    return response