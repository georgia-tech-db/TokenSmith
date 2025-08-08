"""
Prompt builder + llama.cpp wrapper (adapted from dbchat snippet).
"""
import subprocess, shlex, textwrap
from typing import List

LLAMA_CPP_BINARY = "/Users/aj/git/llama.cpp/build/bin/llama-cli"

def format_prompt(chunks: List[str], query: str, max_chunk_chars: int = 500) -> str:
    trimmed = [c[:max_chunk_chars] for c in chunks]
    context = "\n\n".join(trimmed)
    return textwrap.dedent(f"""\
        <|im_start|>system
        You are a helpful assistant. Use the following textbook excerpts to answer the question.
        <|im_end|>
        <|im_start|>user
        Textbook Excerpts:
        {context}

        Question: {query}
        <|im_end|>
        <|im_start|>assistant
    """)

def run_llama_cpp(prompt: str,
                  model_path: str,
                  max_tokens: int = 500,
                  extra_args: str = "-no-cnv") -> str:
    cmd = f'{LLAMA_CPP_BINARY} -m {shlex.quote(model_path)} -p {shlex.quote(prompt)} -n {max_tokens} {extra_args}'
    proc = subprocess.Popen(shlex.split(cmd),
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True)
    output = []
    for line in proc.stdout:
        print(line, end="")          # stream to console
        output.append(line)
    proc.wait()
    return "".join(output)

def answer(query: str,
           chunks: List[str],
           model_path: str,
           max_tokens: int = 500) -> str:
    prompt = format_prompt(chunks, query)
    print(f"\n⚙️  Prompt length = {len(prompt.split())} tokens (approx)\n")
    return run_llama_cpp(prompt, model_path, max_tokens)
