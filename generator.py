import os, subprocess, textwrap

LLAMA_CPP_BINARY = os.getenv("LLAMA_CPP_BIN", "/Users/aj/git/llama.cpp/build/bin/llama-cli")

ANSWER_START = "<<<ANSWER>>>"
ANSWER_END   = "<<<END>>>"

def format_prompt(chunks, query, max_chunk_chars=400):
    # smaller chunks = less repetition fuel
    trimmed = [(c or "")[:max_chunk_chars] for c in chunks]
    context = "\n\n".join(trimmed)
    return textwrap.dedent(f"""\
        <|im_start|>system
        You are a concise tutor. Use the textbook excerpts to answer in 2–3 sentences.
        If the excerpts are insufficient, say so briefly.
        End your reply with {ANSWER_END}.
        <|im_end|>
        <|im_start|>user
        Textbook Excerpts:
        {context}

        Question: {query}
        <|im_end|>
        <|im_start|>assistant
        {ANSWER_START}
    """)

def _extract_answer(raw: str) -> str:
    # take everything after the last START marker, then cut at END
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
        #"--ngl", str(n_gpu_layers),
        "--temp", str(temperature),
        "--top-k", "40",
        "--top-p", "0.9",
        "--min-p", "0.05",
        "--typical", "1.0",
        "--repeat-penalty", "1.15",
        "--repeat-last-n", "256",
        "--mirostat", "2",              # stabilizes length/quality
        "--mirostat-ent", "5",
        "--mirostat-lr", "0.1",
        "--no-mmap",
        "-no-cnv",
        "-r", ANSWER_END,               # hard stop at <<<END>>>
    ]
    # capture BOTH streams (some builds stream tokens on stderr)
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
    # simple consecutive-sentence de-dupe
    import re
    sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    cleaned = []
    for s in sents:
        if not cleaned or s.lower() != cleaned[-1].lower():
            cleaned.append(s)
    return " ".join(cleaned)

def answer(query: str, chunks, model_path: str, max_tokens: int = 300, **kw):
    prompt = format_prompt(chunks, query)
    approx_tokens = max(1, len(prompt) // 4)
    print(f"\n⚙️  Prompt length ≈ {approx_tokens} tokens\n")
    raw = run_llama_cpp(prompt, model_path, max_tokens=max_tokens, **kw)
    return _dedupe_sentences(raw)
