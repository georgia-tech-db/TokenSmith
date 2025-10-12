#!/usr/bin/env python3
"""
preprocess.py
PDF → text → chunks (via DocumentChunker + strategy) → embeddings → FAISS + metadata

Entry point (called by main.py):
    build_index(markdown_file, out_prefix, cfg, keep_tables=True, do_visualize=False)
"""

import os
import pickle
import re
from typing import List, Dict

import faiss
import nltk
from sentence_transformers import SentenceTransformer

from src.config import QueryPlanConfig
from src.chunking import DocumentChunker
from src.extracting.extraction import chunk_markdown_by_headings

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

TABLE_RE = re.compile(r"<table>.*?</table>", re.DOTALL | re.IGNORECASE)

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

def build_index(
    markdown_file: str,
    out_prefix: str = "textbook_index",
    *,
    cfg: QueryPlanConfig,
    keep_tables: bool = True,
    do_visualize: bool = False,
) -> None:
    """
    Extract sections from markdown, chunk via strategy, embed, build FAISS, and persist:
        index/{strategy_folder}/{out_prefix}.faiss
        index/{strategy_folder}/{out_prefix}_chunks.pkl
        index/{strategy_folder}/{out_prefix}_sources.pkl
        index/{strategy_folder}/{out_prefix}_meta.pkl
    """
    all_chunks: List[str] = []
    sources: List[str] = []
    metadata: List[Dict] = []

    # Extract sections from markdown
    sections = chunk_markdown_by_headings(markdown_file)

    # Create strategy and chunker
    strategy = cfg.make_strategy()
    chunker = DocumentChunker(strategy=strategy, keep_tables=keep_tables)

    # 1) Chunk using DocumentChunker
    for i, c in enumerate(sections):
        has_table = bool(TABLE_RE.search(c['content']))
        meta = {
            "filename": markdown_file,
            "chunk_id": i,
            "mode": cfg.chunk_config.to_string(),
            "keep_tables": keep_tables,
            "char_len": len(c['content']),
            "word_len": len(c['content'].split()),
            "has_table": has_table,
            "section": c['heading'], 
            "text_preview": c['content'][:100]
        }
        
        # Use DocumentChunker to recursively split this section
        sub_chunks = chunker.chunk(c['content'])
        for sub_c in sub_chunks:
            all_chunks.append(sub_c)
            sources.append(markdown_file)
            metadata.append(meta)

    # 2) embed
    print(f"🪄  Embedding {len(all_chunks):,} chunks with {cfg.embed_model} …")
    embedder = SentenceTransformer(cfg.embed_model)
    embeddings = embedder.encode(
        all_chunks, batch_size=4, show_progress_bar=True
    ).astype("float32")

    # 3) FAISS
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # 4) Build prefixes based on strategy
    faiss_prefix = cfg.get_faiss_prefix(out_prefix)

    # 5) persist
    faiss.write_index(index, f"{faiss_prefix}.faiss")
    with open(f"{faiss_prefix}_chunks.pkl", "wb") as f:
        pickle.dump(all_chunks, f)
    with open(f"{faiss_prefix}_sources.pkl", "wb") as f:
        pickle.dump(sources, f)
    with open(f"{faiss_prefix}_meta.pkl", "wb") as f:
        pickle.dump(metadata, f)

    print(f"✓ Index built: {faiss_prefix}.faiss  |  {len(all_chunks)} chunks")

    # 6) optional viz
    if do_visualize:
        visualize(embeddings, sources)

def visualize(embeddings, sources):
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
