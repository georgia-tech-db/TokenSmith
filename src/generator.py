import textwrap, re
from llama_cpp import Llama, LlamaRAMCache

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

def get_system_prompt(mode="new_v1"):
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
        "new_v1" : textwrap.dedent(f"""
            You are an expert, encouraging academic tutor. Your goal is to help the student understand the material deeply. Follow these execution steps:

            1. ANALYZE: Break down the student's question to identify all core concepts and sub-questions that need answering.
            2. GROUNDING: Refer PRIMARY to the provided textbook excerpts to formulate your answer. 
            - If the excerpts do not contain the answer, please state that the question cannot be answered from the provided material.
            - If no excerpts are provided (this means 'your knowledge mode'), answer accurately using your general knowledge.
            3. STYLE: Explain the concepts clearly and step-by-step, adapting your tone to be supportive and educational. 
            4. ENGAGEMENT: Try to avoid giving too many examples or analogies and focus instead on answering the question directly and clearly. Do not drift into unrelated tangents or topics.
            5. CLARITY: Avoid fluff. Focus on giving a clear answer to the question and stay within its scope.

            End your reply with {ANSWER_END}.
        """).strip(),

        "tutor": textwrap.dedent(f"""
            You are a tutor. Follow these rules:
            1. Analyze the user question and identify all parts that need answering.
            2. Refer ONLY to the provided textbook excerpts to find answers to all parts if they are provided.
            3. Answer the question completely and concisely, as if teaching a student.
            4. Refrain from giving examples and analgogies unless they are explicitly requested.
            5. Be concise and clear and only answer the question asked without drifting into unrelated topics.
            End your reply with {ANSWER_END}.
        """).strip(),
        
        "concise": textwrap.dedent(f"""
            You are a concise assistant. Answer questions briefly and directly using the provided textbook excerpts when they are provided.
            - Keep answers short and to the point.
            - Focus on key concepts only.
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
        if isinstance(chunks[0], tuple):
            chunks = [c[0] for c in chunks]
        context = "\n\n".join(chunks)
        context = text_cleaning(context)
        
        # Build prompt with chunks
        context_section = f"Textbook Excerpts:\n{context}\n\n\n"
        
        final_prompt = textwrap.dedent(f"""\
            {system_section}<|im_start|>user
            {context_section}Question: {query}
            <|im_end|>
            <|im_start|>assistant
            {ANSWER_START}
        """)

        # print("Formatted prompt with chunks. Length:", len(final_prompt), final_prompt)
        return final_prompt

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

_LLM_CACHE = {}

def get_llama_model(model_path: str, n_ctx: int = 8192):
    if model_path not in _LLM_CACHE:
        try:
            _LLM_CACHE[model_path] = Llama(model_path=model_path,
                                       n_ctx=n_ctx,
                                       verbose=False,
                                       n_gpu_layers=-1,
                                       flash_attn=True)
        except Exception as e:
            print(f"Error loading LLaMA model from {model_path} on GPU: {e}")
            _LLM_CACHE[model_path] = Llama(model_path=model_path,
                                       n_ctx=n_ctx,
                                       verbose=False)

        cache = LlamaRAMCache()
        _LLM_CACHE[model_path].set_cache(cache)
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

def answer(query: str, chunks, model_path: str, max_tokens: int = 300, system_prompt_mode: str = "tutor", temperature: float = 0.2):
    prompt = format_prompt(chunks, query, system_prompt_mode=system_prompt_mode)
    return stream_llama_cpp(prompt, model_path, max_tokens=max_tokens, temperature=temperature)

def double_answer(query: str, chunks, model_path: str,
                  max_tokens: int = 300,
                  system_prompt_mode: str = "tutor",
                  temperature: float = 0.2):

    # ---- Pass 1 ----
    base_prompt = format_prompt(
        chunks,
        query,
        system_prompt_mode=system_prompt_mode
    )

    initial_stream = stream_llama_cpp(
        base_prompt,
        model_path,
        max_tokens,
        temperature
    )

    initial_response = "".join(initial_stream)
    initial_response = dedupe_generated_text(initial_response)

    # ---- Pass 2 (repeat SAME question) ----
    repeated_prompt = (
        base_prompt
        + initial_response
        + f"\n{ANSWER_END}\n"
        + "<|im_end|>\n"
        + "<|im_start|>user\n"
        + f"Question: {query}"
        + "\n<|im_end|>\n"
        + "<|im_start|>assistant\n"
        + ANSWER_START
    )

    return stream_llama_cpp(
        repeated_prompt,
        model_path,
        max_tokens,
        temperature
    )

def dedupe_generated_text(text: str) -> str:
    """
    Removes immediate consecutive duplicate sentences or lines from LLM output.
    Keeps Markdown/code formatting intact.
    """
    lines = text.split("\n")
    cleaned = []
    prev = None
    for line in lines:
        normalized = line.strip().lower()
        # Skip if this line is a repeat of the previous one
        if normalized == prev and normalized != "":
            continue
        cleaned.append(line)
        prev = normalized
    return "\n".join(cleaned)