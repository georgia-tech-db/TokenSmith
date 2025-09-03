"""
preprocess.py
PDF → text → chunks (table-safe) → embeddings → FAISS + metadata

Entry point (called by main.py):
    build_index(pdf_dir, out_prefix, model_name, chunk_size_char,
                chunk_mode="chars", chunk_tokens=500, keep_tables=True,
                pdf_range=None, pdf_files=None, do_visualize=False)
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from typing import List, Dict, Tuple, Optional
import re
import pickle
import pathlib

import fitz  # PyMuPDF
import faiss
import numpy as np
from tqdm import tqdm

import nltk
from nltk.tokenize import sent_tokenize
nltk.download("punkt", quiet=True)

from sentence_transformers import SentenceTransformer


# ----------------------------- Chunker ---------------------------------

class DocumentChunker:
    """
    Chunk text by sentences (token-aware) or raw chars, with an option to
    preserve <table>...</table> blocks so they never straddle boundaries.
    """

    TABLE_RE = re.compile(r"<table>.*?</table>", re.DOTALL | re.IGNORECASE)

    def __init__(
        self,
        max_chars: int = 20_000,
        mode: str = "chars",            # "chars" | "tokens"
        max_tokens: int = 500,          # used only if mode == "tokens"
        keep_tables: bool = True,
    ):
        self.max_chars = max_chars
        self.mode = mode
        self.max_tokens = max_tokens
        self.keep_tables = keep_tables

    def _extract_tables(self, text: str) -> Tuple[str, List[str]]:
        tables = self.TABLE_RE.findall(text)
        for i, t in enumerate(tables):
            text = text.replace(t, f"[TABLE_PLACEHOLDER_{i}]")
        return text, tables

    @staticmethod
    def _restore_tables(chunk: str, tables: List[str]) -> str:
        for i, t in enumerate(tables):
            ph = f"[TABLE_PLACEHOLDER_{i}]"
            if ph in chunk:
                chunk = chunk.replace(ph, t)
        return chunk

    def chunk(self, text: str) -> List[str]:
        if not text:
            return []

        work = text
        tables: List[str] = []
        if self.keep_tables:
            work, tables = self._extract_tables(work)

        if self.mode == "tokens":
            # sentence-aware; bound by approx word-count
            chunks: List[str] = []
            cur, cur_len = [], 0
            for s in sent_tokenize(work):
                w = len(s.split())
                if cur and cur_len + w > self.max_tokens:
                    chunks.append(" ".join(cur))
                    cur, cur_len = [s], w
                else:
                    cur.append(s)
                    cur_len += w
            if cur:
                chunks.append(" ".join(cur))
        elif self.mode == "chars":
            # naive char slicing
            step = self.max_chars
            chunks = [work[i:i + step] for i in range(0, len(work), step)]
        elif self.mode == "sections":
            # section-wise slicing
            heading_re = re.compile(r"""
                (?m)
                ^(?=.{,120}$)\s{0,3}
                (?P<num>[1-9]\d*\.[0-9]+(?:\.[0-9]+)*)   # requires at least one dot
                (?![)\]])
                \s+(?P<title>(?!\d).+?)\s*$
                """, re.VERBOSE)
            matches = list(heading_re.finditer(text))

            for i in range(100):
                print("i: "+str(i)+ matches[i].group(0))

            heads = []
            for m in matches:
                num = m.group("num")
                title = m.group("title").strip()
                level = num.count(".") + 1
                heads.append({
                    "num": num, "title": title, "level": level,
                    "start": m.start(), "endline": m.end()
                })

            chunks = []
            N = len(heads)
            for i, h in enumerate(heads):
                end_idx = len(text)
                for j in range(i + 1, N):
                    if heads[j]["level"] <= h["level"]:
                        end_idx = heads[j]["start"]
                        break

                body = text[h["endline"]:end_idx].strip("\n").strip()
                ### for referencing section numbers in the future
                ### ignoring for now, focusing on search
                # chunks.append({
                #     "id": h["num"],
                #     "title": h["title"],
                #     "level": h["level"],
                #     "text": body
                # })
                chunks.append(body)

        if self.keep_tables and tables:
            chunks = [self._restore_tables(c, tables) for c in chunks]
        return chunks


# ------------------------ Section Guessing (metadata) -------------------

SECTION_RE = re.compile(
    r"^\s*(Chapter\s+\d+|Section\s+\d+(?:\.\d+)*|[A-Z][A-Za-z0-9\s\-]{3,})",
    re.MULTILINE,
)

def guess_section_headers(text: str, max_headers: int = 50) -> List[str]:
    """Heuristic section headers for Segment/Filter hints."""
    hits = SECTION_RE.findall(text)
    headers = [h[0].strip() if isinstance(h, tuple) else str(h).strip() for h in hits]
    return headers[:max_headers] if headers else []


# -------------------------- Core pipeline ------------------------------

def _resolve_pdf_paths(
    pdf_dir: str,
    pdf_range: Optional[Tuple[int, int]],
    pdf_files: Optional[List[str]],
) -> List[pathlib.Path]:
    base = pathlib.Path(pdf_dir)
    if pdf_files:
        paths = [base / name for name in pdf_files]
    elif pdf_range:
        start, end = pdf_range
        wanted = {f"{i}.pdf" for i in range(start, end)}
        paths = [p for p in base.glob("*.pdf") if p.name in wanted]
        paths.sort()
    else:
        paths = sorted(base.glob("*.pdf"))
    return paths


def build_index(
    pdf_dir: str,
    out_prefix: str = "textbook_index",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    chunk_size_char: int = 20_000,
    *,
    chunk_mode: str = "chars",                 # "tokens" | "chars"
    chunk_tokens: int = 500,                   # used if chunk_mode == "tokens"
    keep_tables: bool = True,
    pdf_range: Optional[Tuple[int, int]] = None,  # e.g., (27, 33)
    pdf_files: Optional[List[str]] = None,        # e.g., ["27.pdf","28.pdf"]
    do_visualize: bool = False,
) -> None:
    """
    Extract PDFs from *pdf_dir*, chunk (table-safe), embed, build FAISS, and persist:
        {out_prefix}.faiss
        {out_prefix}_chunks.pkl
        {out_prefix}_sources.pkl
        {out_prefix}_meta.pkl
    """
    pdf_paths = _resolve_pdf_paths(pdf_dir, pdf_range, pdf_files)
    if not pdf_paths:
        raise FileNotFoundError(
            f"No PDFs found in {pdf_dir} (range={pdf_range}, files={pdf_files})"
        )

    # 1) extract + chunk (collect metadata)
    chunker = DocumentChunker(
        max_chars=chunk_size_char,
        mode=chunk_mode,
        max_tokens=chunk_tokens,
        keep_tables=keep_tables,
    )

    all_chunks: List[str] = []
    sources: List[str] = []
    metadata: List[Dict] = []

    for path in tqdm(pdf_paths, desc="⛏️  extracting PDFs"):
        with fitz.open(path) as doc:
            full_text = "".join(page.get_text() for page in doc)
            #print(full_text)

        headers = guess_section_headers(full_text)
        chunks = chunker.chunk(full_text)

        for i, c in enumerate(chunks):
            has_table = bool(DocumentChunker.TABLE_RE.search(c))
            all_chunks.append(c)
            sources.append(path.name)
            metadata.append(
                {
                    "filename": path.name,
                    "chunk_id": i,
                    "mode": chunk_mode,
                    "keep_tables": keep_tables,
                    "char_len": len(c),
                    "word_len": len(c.split()),
                    "has_table": has_table,
                    "section_hints": headers[:10],  # small header sample
                }
            )

    # 2) embed
    print(f"🪄  Embedding {len(all_chunks):,} chunks with {model_name} …")
    embedder = SentenceTransformer(model_name, device="cpu")
    embeddings = embedder.encode(
        all_chunks, batch_size=4, show_progress_bar=True
    ).astype("float32")

    # 3) FAISS
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # 4) persist
    faiss.write_index(index, f"{out_prefix}.faiss")
    with open(f"{out_prefix}_chunks.pkl", "wb") as f:
        pickle.dump(all_chunks, f)
    with open(f"{out_prefix}_sources.pkl", "wb") as f:
        pickle.dump(sources, f)
    with open(f"{out_prefix}_meta.pkl", "wb") as f:
        pickle.dump(metadata, f)

    print(f"✓ Index built: {out_prefix}.faiss  |  {len(all_chunks)} chunks")

    # 5) optional viz
    if do_visualize:
        try:
            from sklearn.decomposition import PCA
            import matplotlib.pyplot as plt

            red = PCA(n_components=2).fit_transform(embeddings)
            uniq = sorted(set(sources))
            cmap = {s: i for i, s in enumerate(uniq)}
            colors = [cmap[s] for s in sources]

            plt.figure(figsize=(10, 7))
            sc = plt.scatter(
                red[:, 0], red[:, 1], c=colors, cmap="tab10", alpha=0.55
            )
            plt.title("Vector index (PCA)")
            plt.legend(
                handles=sc.legend_elements()[0],
                labels=uniq,
                bbox_to_anchor=(1.02, 1),
                loc="upper left",
            )
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"[visualize] skipped ({e})")
