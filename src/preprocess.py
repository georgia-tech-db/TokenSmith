#!/usr/bin/env python3
"""
preprocess.py
PDF → text → chunks (table-safe) → embeddings → FAISS + metadata

Entry point (called by main.py):
    build_index(pdf_dir, out_prefix, model_name, chunk_size_char,
                chunk_mode="chars", chunk_tokens=500, keep_tables=True,
                pdf_range=None, pdf_files=None, do_visualize=False)
"""

import os
import re
import pickle
import pathlib
from typing import List, Dict, Tuple, Optional

import fitz  # PyMuPDF
import faiss
from tqdm import tqdm
import nltk
from sentence_transformers import SentenceTransformer

from src.chunking import ChunkStrategy, SlidingTokenStrategy
from src.config import QueryPlanConfig
from src.ranking.tagging import build_tfidf_tags

# ----- runtime parallelism knobs (avoid oversubscription) -----
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# NLTK resources (quiet)
nltk.download("punkt", quiet=True)
# Newer NLTK may require punkt_tab; safe to attempt quietly
try:
    nltk.download("punkt_tab", quiet=True)
except Exception:
    pass


# ----------------------------- Chunker ---------------------------------

class DocumentChunker:
    """
    Chunk text by sentences/tokens or raw chars via a provided strategy,
    with an optional special mode "sections" that slices by numeric headings.
    Table blocks (<table>...</table>) are preserved within chunks.
    """

    TABLE_RE = re.compile(r"<table>.*?</table>", re.DOTALL | re.IGNORECASE)

    def __init__(
        self,
        strategy: Optional[ChunkStrategy],
        keep_tables: bool = True
    ):
        if mode != "tokens" and mode != "chars" and mode != "sections":
            raise ValueError("Invalid mode provided. Must be 'tokens', 'chars', or 'sections'")

        self.strategy = strategy
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

        if self.strategy is None:
            # Defensive fallback: if no strategy provided, return as one chunk
            chunks = [work]
        else:
            chunks = self.strategy.chunk(work)

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
    *,
    cfg: QueryPlanConfig,
    keep_tables: bool = True,
    pdf_range: Optional[Tuple[int, int]] = None,  # e.g., (27, 33)
    pdf_files: Optional[List[str]] = None,        # e.g., ["27.pdf","28.pdf"]
    do_visualize: bool = False,
) -> None:
    """
    Extract PDFs from *pdf_dir*, chunk (table-safe), embed, build FAISS, and persist:
        index/{chunking_strategy_folder}/{out_prefix}.faiss
        index/{chunking_strategy_folder}/{out_prefix}_chunks.pkl
        index/{chunking_strategy_folder}/{out_prefix}_sources.pkl
        index/{chunking_strategy_folder}/{out_prefix}_meta.pkl
    """

    # 1) extract + chunk (collect metadata)
    strategy: ChunkStrategy = cfg.make_strategy()
    chunker = DocumentChunker(strategy, keep_tables=keep_tables)

    pdf_paths = _resolve_pdf_paths(pdf_dir, pdf_range, pdf_files)
    if not pdf_paths:
        raise FileNotFoundError(
            f"No PDFs found in {pdf_dir} (range={pdf_range}, files={pdf_files})"
        )

    all_chunks: List[str] = []
    sources: List[str] = []
    metadata: List[Dict] = []

    for path in tqdm(pdf_paths, desc="⛏️  extracting PDFs"):
        with fitz.open(path) as doc:
            full_text = "".join(page.get_text() for page in doc)

        headers = guess_section_headers(full_text)
        chunks = chunker.chunk(full_text)

        for i, c in enumerate(chunks):
            has_table = bool(DocumentChunker.TABLE_RE.search(c))
            meta = {
                "filename": path.name,
                "chunk_id": i,
                "mode": cfg.chunk_config.to_string(),
                "keep_tables": keep_tables,
                "char_len": len(c),
                "word_len": len(c.split()),
                "has_table": has_table,
                "section_hints": headers[:10],  # small header sample
            }
            if isinstance(strategy, SlidingTokenStrategy):
                meta["max_tokens"] = strategy.max_tokens
                meta["overlap_tokens"] = strategy.overlap_tokens
                meta["tokenizer_name"] = strategy.tokenizer_name

            all_chunks.append(c)
            sources.append(path.name)
            metadata.append(meta)

    # 2) Tag the chunks (offline)
    print("🏷️  Building TF-IDF tags …")
    vectorizer, chunk_tags = build_tfidf_tags(
        all_chunks,
        ngram_range=(1, 3),
        max_features=25000,
        min_df=2,
        max_df=0.6,
        top_k_per_chunk=10,
    )
    for i, tags in enumerate(chunk_tags):
        metadata[i]["tags"] = tags

    # 3) embed
    print(f"🪄  Embedding {len(all_chunks):,} chunks with {cfg.embed_model} …")
    embedder = SentenceTransformer(cfg.embed_model, device="cpu")
    embeddings = embedder.encode(
        all_chunks, batch_size=4, show_progress_bar=True
    ).astype("float32")

    # 4) FAISS
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # 5) Build prefixes based on chunking strategy
    faiss_prefix = cfg.get_faiss_prefix(out_prefix)
    meta_prefix = cfg.get_tfidf_prefix(out_prefix)

    # 6) persist
    faiss.write_index(index, f"{faiss_prefix}.faiss")
    with open(f"{faiss_prefix}_chunks.pkl", "wb") as f:
        pickle.dump(all_chunks, f)
    with open(f"{faiss_prefix}_sources.pkl", "wb") as f:
        pickle.dump(sources, f)
    with open(f"{faiss_prefix}_meta.pkl", "wb") as f:
        pickle.dump(metadata, f)

    # persist tagging artifacts under meta/
    os.makedirs("meta", exist_ok=True)
    with open(f"{meta_prefix}_tfidf.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open(f"{meta_prefix}_tags.pkl", "wb") as f:
        pickle.dump(chunk_tags, f)

    print(f"✓ Index built: {faiss_prefix}.faiss  |  {len(all_chunks)} chunks")

    # 6) optional viz
    if do_visualize:
        try:
            from sklearn.decomposition import PCA
            import matplotlib.pyplot as plt

            red = PCA(n_components=2).fit_transform(embeddings)
            uniq = sorted(set(sources))
            cmap = {s: i for i, s in enumerate(uniq)}
            colors = [cmap[s] for s in sources]

            plt.figure(figsize=(10, 7))
            sc = plt.scatter(red[:, 0], red[:, 1], c=colors, cmap="tab10", alpha=0.55)
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
