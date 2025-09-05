import re
import textwrap
from typing import Optional
import fitz  # PyMuPDF
from tqdm import tqdm

from src.preprocess import DocumentChunker, _resolve_pdf_paths, guess_section_headers
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
    header = text_cleaning(header)
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
    model_path: str,
    pdf_dir: str,
    pdf_range: Optional[tuple[int, int]] = None,  # e.g., (27, 33)
    pdf_files: Optional[list[str]] = None,  # e.g., ["27.pdf","28.pdf"]):
):
    chunker = DocumentChunker(None, keep_tables=True, mode="section")

    pdf_paths = _resolve_pdf_paths(pdf_dir, pdf_range, pdf_files)
    if not pdf_paths:
        raise FileNotFoundError(
            f"No PDFs found in {pdf_dir} (range={pdf_range}, files={pdf_files})"
        )

    for path in tqdm(pdf_paths, desc="⛏️  extracting PDFs"):
        with fitz.open(path) as doc:
            full_text = "".join(page.get_text() for page in doc)

    chunks = chunker.chunk(full_text)

    with open("summary_index.txt", "w") as f:
        for chunk in chunks:
            query = summary_prompt(chunk)
            summary = run_llama_cpp(query, model_path)
            f.write(summary + "\n")
