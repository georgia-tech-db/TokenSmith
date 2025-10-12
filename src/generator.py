import os, subprocess, textwrap, re, shutil, pathlib

ANSWER_START = "<<<ANSWER>>>"
ANSWER_END   = "<<<END>>>"

def _project_root() -> pathlib.Path:
    # generator.py is in src/, so project root is parent of that folder
    here = pathlib.Path(__file__).resolve()
    return here.parent.parent

def _read_llama_pathfile() -> str | None:
    pathfile = _project_root() / "src" / "llama_path.txt"
    try:
        p = pathfile.read_text(encoding="utf-8").strip()
        return p or None
    except FileNotFoundError:
        return None

def _is_executable(p: str | os.PathLike) -> bool:
    return p and os.path.isfile(p) and os.access(p, os.X_OK)

def resolve_llama_binary() -> str:
    """
    Resolution order:
      1) $LLAMA_CPP_BINARY (absolute or name on PATH)
      2) src/llama_path.txt (written by build_llama.sh)
      3) 'llama-cli' on PATH
    Raises a helpful error if none work.
    """
    # 1) Env var
    env_bin = os.getenv("LLAMA_CPP_BINARY")
    if env_bin:
        if _is_executable(env_bin):
            return env_bin
        found = shutil.which(env_bin)
        if found:
            return found

    # 2) Path file from build script
    file_bin = _read_llama_pathfile()
    if file_bin and _is_executable(file_bin):
        return file_bin

    # 3) PATH
    path_bin = shutil.which("llama-cli")
    if path_bin:
        return path_bin

    # No dice → explain how to fix
    raise FileNotFoundError(
        "Could not locate 'llama-cli'. Tried $LLAMA_CPP_BIN, src/llama_path.txt, and PATH.\n"
        "Fixes:\n"
        "  • Run:  make build-llama   (writes src/llama_path.txt)\n"
        "  • Or set:  export LLAMA_CPP_BIN=/absolute/path/to/llama-cli\n"
        "  • Or install llama.cpp and ensure 'llama-cli' is on your PATH."
    )

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

def format_prompt(chunks, query, max_chunk_chars=400):
    trimmed = [(c or "")[:max_chunk_chars] for c in chunks]
    context = "\n\n".join(trimmed)
    context = text_cleaning(context)
    return textwrap.dedent(f"""\
        <|im_start|>system
        The user is currently studying. They have asked a query and I have provided a set of relevant textbook excerpts below. Use only those excerpts to answer if they have been provided. If the excerpts are empty, say "I don't know".
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
    text = raw.split(ANSWER_START)[-1]
    return text.split(ANSWER_END)[0].strip()

def run_llama_cpp(prompt: str, model_path: str, max_tokens: int = 300,
                  threads: int = 8, n_gpu_layers: int = 8, temperature: float = 0.3):
    llama_binary = resolve_llama_binary()
    cmd = [
        llama_binary,
        "-m", model_path,
        "-p", prompt,
        "-n", str(max_tokens),
        "-t", str(threads),
        # "--ngl", str(n_gpu_layers),
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
