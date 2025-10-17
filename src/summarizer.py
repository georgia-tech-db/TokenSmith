import textwrap
from typing import Optional
import fitz  # PyMuPDF
from tqdm import tqdm
import sys
import os
import pathlib

from src.utils import text_cleaning

src_module = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(str(src_module))
sys.path.append(str(src_module.parent))

from src.preprocessing.chunking import DocumentChunker
from src.preprocessing.chunking import SectionRecursiveStrategy, SectionRecursiveConfig
from src.generator import run_llama_cpp

ANSWER_START = "<<<ANSWER>>>"
ANSWER_END = "<<<END>>>"


def summary_prompt(section: str) -> str:
    section = text_cleaning(section)
    return textwrap.dedent(
        f"""\
        <|im_start|>system
        You are a textbook summarizer. Your job is to summarize the following section of a Databases textbook in a couple sentences
        while retaining conceptual information
        and important definitions. \
        The summary must be shorter than the original section.
        End your reply with {ANSWER_END}.
        <|im_end|>
        <|im_start|>user
        
        Textbook Section:
        {section}

        <|im_end|>
        <|im_start|>assistant
        {ANSWER_START}
    """
    )


def build_summary_index(
    model_path: os.PathLike = "build/llama.cpp/models/qwen2.5-0.5b-instruct-q5_k_m.gguf",
    pdf_dir: str = "data/chapters/",
):
    model_path = pathlib.Path(model_path)
    print(f"Building summary index using model: {model_path}")
    chunker = DocumentChunker(SectionRecursiveStrategy(SectionRecursiveConfig()), keep_tables=True)

    with fitz.open(pathlib.Path(pdf_dir, "silberschatz.pdf")) as doc:
        full_text = "".join(page.get_text() for page in doc)

    chunks = chunker.chunk(full_text)
    print(f"Number of chunks: {len(chunks)}")

    llama_debug_line_prefixes = [
        "llama_perf_sampler_print:",
        "llama_perf_context_print:",
        "llama_model_loader:",
        "llama_model_load_from_file_impl:",
        "ggml_cuda_init:",
        "Device 0:",
        "Device 1:",
        "build:",
        "main:",
        "load:",
        "print_info:",
        "load_tensors:",
        "llama_context:",
        "llama_kv_cache:",
        "common_init_from_params:",
        "system_info:",
        ".........",
        "<think>",
        "</think>",
    ]

    def is_debug_line(line: str) -> bool:
        stripped_line = line.strip()

        if stripped_line == "Summary:":
            return True

        for prefix in llama_debug_line_prefixes:
            if stripped_line.startswith(prefix):
                return True

        return False

    with open(f"summary_index-{model_path.stem}.txt", "w") as f:
        for chunk in tqdm(chunks):
            query = summary_prompt(chunk)
            response = run_llama_cpp(query, model_path)
            response_lines = response.split("\n")
            answer_lines = [
                f"{r_line}\n"
                for r_line in response_lines
                if len(r_line) > 0 and not is_debug_line(r_line)
            ]
            f.writelines(answer_lines)

def main():
    model_path = pathlib.Path("build", "llama.cpp", "models", "Qwen3-1.7B-Q8_0.gguf")
    build_summary_index(model_path=model_path)

if __name__ == "__main__":
    main()
