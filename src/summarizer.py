import re
import textwrap
from typing import Optional
import fitz  # PyMuPDF
from tqdm import tqdm
import sys
import os
import pathlib

src_module = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(str(src_module))
sys.path.append(str(src_module.parent))

from src.preprocess import DocumentChunker, _resolve_pdf_paths
from src.generator import run_llama_cpp

ANSWER_START = "<<<ANSWER>>>"
ANSWER_END = "<<<END>>>"


def text_cleaning(prompt):
    _CONTROL_CHARS_RE = re.compile(r"[\u0000-\u001F\u007F-\u009F]")
    _DANGEROUS_PATTERNS = [
        r"ignore\s+(all\s+)?previous\s+instructions?",
        r"you\s+are\s+now\s+(in\s+)?developer\s+mode",
        r"system\s+override",
        r"reveal\s+prompt",
    ]
    text = _CONTROL_CHARS_RE.sub("", prompt)
    text = re.sub(r"\s+", " ", text).strip()
    for pat in _DANGEROUS_PATTERNS:
        text = re.sub(pat, "[FILTERED]", text, flags=re.IGNORECASE)
    return text


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
    model_path: str = "models/qwen2.5-0.5b-instruct-q5_k_m.gguf",
    pdf_dir: str = "../data/chapters/",
    pdf_range: Optional[tuple[int, int]] = None,  # e.g., (27, 33)
    pdf_files: Optional[list[str]] = None,  # e.g., ["27.pdf","28.pdf"]):
):
    chunker = DocumentChunker(None, keep_tables=True, mode="section")


    with fitz.open(pathlib.Path(pdf_dir, "silberschatz.pdf")) as doc:
        full_text = "".join(page.get_text() for page in doc)

    chunks = chunker.chunk(full_text)

    with open("summary_index.txt", "w") as f:
        for chunk in chunks:
            query = summary_prompt(chunk)
            summary = run_llama_cpp(query, model_path)
            f.write(summary + "\n")


if __name__ == "__main__":
    build_summary_index()
