import os, subprocess, textwrap

LLAMA_CPP_BINARY = os.getenv("LLAMA_CPP_BIN", "/Users/Priya/Documents/Personal Projects/TokenSmith/llama.cpp/build/bin/llama-cli")

ANSWER_START = "<<<ANSWER>>>"
ANSWER_END   = "<<<END>>>"

def format_prompt(chunks, query, max_chunk_chars=400, prompt_style="default"):
    trimmed = [(c or "")[:max_chunk_chars] for c in chunks]
    context = "\n\n".join(trimmed)
    

    prompt_templates = {
        "default": f"""\
        <|im_start|>system
        You are a concise tutor. Use the textbook excerpts to answer in 2–3 sentences.
        If the excerpts are insufficient, say so briefly.
        End your reply with {ANSWER_END}.
        <|im_end|>""",
        
        "detailed": f"""\
        <|im_start|>system
        You are a comprehensive tutor. Provide detailed explanations using the textbook excerpts.
        Include examples and context to make your answer complete and thorough.
        Aim for 4-6 sentences to ensure completeness.
        End your reply with {ANSWER_END}.
        <|im_end|>""",
        
        "simple": f"""\
        <|im_start|>system
        You are a clear and simple tutor. Use the textbook excerpts to give straightforward answers.
        Avoid technical jargon and complex explanations. Keep it simple and easy to understand.
        End your reply with {ANSWER_END}.
        <|im_end|>""",
        
        "focused": f"""\
        <|im_start|>system
        You are a focused tutor. Answer the specific question asked using the textbook excerpts.
        Stay on topic and avoid rambling. Be precise and direct.
        End your reply with {ANSWER_END}.
        <|im_end|>"""
    }
    
    system_prompt = prompt_templates.get(prompt_style, prompt_templates["default"])
    
    return textwrap.dedent(f"""\
        {system_prompt}
        <|im_start|>user
        Textbook Excerpts:
        {context}

        Question: {query}
        <|im_end|>
        <|im_start|>assistant
        {ANSWER_START}
    """)

def _extract_answer(raw: str) -> str:
    text = raw.split(ANSWER_START)[-1]
    return text.split(ANSWER_END)[0].strip()

def run_llama_cpp(prompt: str, model_path: str, max_tokens: int = 300,
                  threads: int = 8, n_gpu_layers: int = 12, temperature: float = 0.3):
    cmd = [
        LLAMA_CPP_BINARY,
        "-m", model_path,
        "-p", prompt,
        "-n", str(max_tokens),
        "-t", str(threads),

        "--temp", str(temperature),
        "--top-k", "40",
        "--top-p", "0.9",
        "--min-p", "0.05",
        "--typical", "1.0",
        "--repeat-penalty", "1.15",
        "--repeat-last-n", "256",
        "--mirostat", "2",
        "--mirostat-ent", "5",
        "--mirostat-lr", "0.1",
        "--no-mmap",
        "-no-cnv",
        "-r", ANSWER_END,
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env={**os.environ, "GGML_LOG_LEVEL": "ERROR", "LLAMA_LOG_LEVEL": "ERROR"},
    )
    out, _ = proc.communicate()
    return _extract_answer(out or "")

def _dedupe_sentences(text: str) -> str:
    import re
    sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    cleaned = []
    for s in sents:
        if not cleaned or s.lower() != cleaned[-1].lower():
            cleaned.append(s)
    return " ".join(cleaned)

def answer(query: str, chunks, model_path: str, max_tokens: int = 300, prompt_style: str = "default", **kw):
    prompt = format_prompt(chunks, query, prompt_style=prompt_style)
    approx_tokens = max(1, len(prompt) // 4)
    print(f"\nPrompt length ≈ {approx_tokens} tokens (style: {prompt_style})\n")
    raw = run_llama_cpp(prompt, model_path, max_tokens=max_tokens, **kw)
    return _dedupe_sentences(raw)
