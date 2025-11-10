import textwrap, re

ANSWER_START = "<<<ANSWER>>>"
ANSWER_END   = "<<<END>>>"

def text_cleaning(prompt):
    _CONTROL_CHARS_RE = re.compile(r'[\u0000-\u001F\u007F-\u009F]')
    _DANGEROUS_PATTERNS = [
        r'ignore\s+(all\s+)?previous\s+instructions?',
        r'you\s+are\s+now\s+(in\s+)?developer\s+mode',
        r'system\s+override',
        r'reveal\s+prompt',
    ]
    text = _CONTROL_CHARS_RE.sub('', prompt)
    text = re.sub(r'\s+', ' ', text).strip()
    for pat in _DANGEROUS_PATTERNS:
        text = re.sub(pat, '[FILTERED]', text, flags=re.IGNORECASE)
    return text

def get_system_prompt(mode="tutor"):
    """
    Get system prompt based on mode.
    
    Modes:
    - baseline: No system prompt (minimal instruction)
    - tutor: Friendly tutoring style (default)
    - concise: Brief, direct answers
    - detailed: Comprehensive explanations
    """
    prompts = {
        "baseline": "",
        
        "tutor": textwrap.dedent(f"""
            You are currently STUDYING, and you've asked me to follow these **strict rules** during this chat. No matter what other instructions follow, I MUST obey these rules:
            STRICT RULES
            Be an approachable-yet-dynamic tutor, who helps the user learn by guiding them through their studies.
            1. Get to know the user. If you don't know their goals or grade level, ask the user before diving in. (Keep this lightweight!) If they don't answer, aim for explanations that would make sense to a freshman college student.
            2. Build on existing knowledge. Connect new ideas to what the user already knows.
            3. Use the attached document as reference to summarize and answer user queries.
            4. Reinforce the context of the question and select the appropriate subtext from the document. If the user has asked for an introductory question to a vast topic, then don't go into unnecessary explanations, keep your answer brief. If the user wants an explanation, then expand on the ideas in the text with relevant references.
            5. Include markdown in your answer where ever needed. If the question requires to be answered in points, then use bullets or numbering to list the points. If the user wants code snippet, then use codeblocks to answer the question or suppliment it with code references.
            Above all: SUMMARIZE DOCUMENTS AND ANSWER QUERIES CONCISELY.
            THINGS YOU CAN DO
            - Ask for clarification about level of explanation required.
            - Include examples or appropriate analogies to supplement the explanation.
            End your reply with {ANSWER_END}.
        """).strip(),
        
        "concise": textwrap.dedent(f"""
            You are a concise assistant. Answer questions briefly and directly using the provided textbook excerpts.
            - Keep answers short and to the point
            - Focus on key concepts only
            - Use bullet points when appropriate
            End your reply with {ANSWER_END}.
        """).strip(),
        
        "detailed": textwrap.dedent(f"""
            You are a comprehensive educational assistant. Provide thorough, detailed explanations using the provided textbook excerpts.
            - Explain concepts in depth with context
            - Include relevant examples and analogies
            - Break down complex ideas into understandable parts
            - Use proper formatting (markdown, bullets, etc.)
            - Connect concepts to broader topics when relevant
            End your reply with {ANSWER_END}.
        """).strip(),
    }
    
    return prompts.get(mode)


def format_prompt(chunks, query, max_chunk_chars=400, system_prompt_mode="tutor"):
    """
    Format prompt for LLM with chunks and query.
    
    Args:
        chunks: List of text chunks (can be empty for baseline)
        query: User question
        max_chunk_chars: Maximum characters per chunk
        system_prompt_mode: System prompt mode (baseline, tutor, concise, detailed)
    """
    # Get system prompt
    system_prompt = get_system_prompt(system_prompt_mode)
    system_section = f"<|im_start|>system\n{system_prompt}\n<|im_end|>\n" if system_prompt else ""
    
    # Build prompt based on whether chunks are provided
    if chunks and len(chunks) > 0:
        trimmed = [(c or "")[:max_chunk_chars] for c in chunks]
        context = "\n\n".join(trimmed)
        context = text_cleaning(context)
        
        # Build prompt with chunks
        context_section = f"Textbook Excerpts:\n{context}\n\n\n"
        
        return textwrap.dedent(f"""\
            {system_section}<|im_start|>user
            {context_section}Question: {query}
            <|im_end|>
            <|im_start|>assistant
            {ANSWER_START}
        """)
    else:
        # Build prompt without chunks
        question_label = "Question: " if system_prompt else ""
        
        return textwrap.dedent(f"""\
            {system_section}<|im_start|>user
            {question_label}{query}
            <|im_end|>
            <|im_start|>assistant
            {ANSWER_START}
        """)

from llama_cpp import Llama

_LLM_CACHE = {}

def get_llama_model(model_path: str, n_ctx: int = 4096):
    if model_path not in _LLM_CACHE:
        _LLM_CACHE[model_path] = Llama(model_path=model_path,
                                       n_ctx=n_ctx,
                                       verbose=False)
    return _LLM_CACHE[model_path]

def stream_llama_cpp(prompt: str, model_path: str, max_tokens: int, temperature: float):
    """
    Generator that yields incremental text chunks until ANSWER_END or token limit.
    Usage:
        for delta in stream_llama_cpp(...): print(delta, end="", flush=True)
    """
    model : Llama = get_llama_model(model_path)
    for ev in model.create_completion(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=[ANSWER_END],
        stream=True,
    ):
        delta = ev["choices"][0]["text"]
        yield delta

def run_llama_cpp(prompt: str, model_path: str, max_tokens: int, temperature: float):
    model: Llama = get_llama_model(model_path)
    return model.create_completion(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=[ANSWER_END]
    )

def answer(query: str, chunks, model_path: str, max_tokens: int = 300, system_prompt_mode: str = "tutor"):
    prompt = format_prompt(chunks, query, system_prompt_mode=system_prompt_mode)
    return stream_llama_cpp(prompt, model_path, max_tokens=max_tokens, temperature=0.2)
