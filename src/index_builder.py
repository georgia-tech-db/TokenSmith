#!/usr/bin/env python3
"""
index_builder.py
PDF -> markdown text -> chunks -> embeddings -> BM25 + FAISS + metadata

Entry point (called by main.py):
    build_index(markdown_file, cfg, keep_tables=True, do_visualize=False)
"""

import os
import pickle
import pathlib
import re
from typing import List, Dict, Optional

import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.preprocessing.chunking import DocumentChunker, ChunkConfig
from src.preprocessing.extraction import extract_sections_from_markdown
from src.summarizer import generate_section_summaries


# ----- runtime parallelism knobs (avoid oversubscription) -----
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

TABLE_RE = re.compile(r"<table>.*?</table>", re.DOTALL | re.IGNORECASE)

# ------------------------ Main index builder -----------------------------

def build_index(
    markdown_file: str,
    *,
    chunker: DocumentChunker,
    chunk_config: ChunkConfig,
    embedding_model_path: os.PathLike,
    artifacts_dir: os.PathLike,
    index_prefix: str, 
    do_visualize: bool = False,
    generate_summaries: bool = True,
    summary_model_path: Optional[os.PathLike] = None,
) -> None:
    """
    Extract sections, chunk, embed, and build both FAISS and BM25 indexes.
    Optionally generate and index section summaries.

    Persists:
        - {prefix}.faiss
        - {prefix}_bm25.pkl
        - {prefix}_chunks.pkl
        - {prefix}_sources.pkl
        - {prefix}_meta.pkl
        - {prefix}_summaries.pkl (if generate_summaries=True)
        - {prefix}_summaries.faiss (if generate_summaries=True)
    """
    embedding_model_path = pathlib.Path(embedding_model_path)
    all_chunks: List[str] = []
    sources: List[str] = []
    metadata: List[Dict] = []

    # Extract sections from markdown
    sections = extract_sections_from_markdown(markdown_file)

    # Step 1: Chunk using DocumentChunker
    for i, c in enumerate(sections):
        has_table = bool(TABLE_RE.search(c['content']))
        meta = {
            "filename": markdown_file,
            "chunk_id": i,
            "mode": chunk_config.to_string(),
            "keep_tables": chunker.keep_tables,
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

    # Step 2: Create embeddings for FAISS index
    print(f"Embedding {len(all_chunks):,} chunks with {embedding_model_path.stem} ...")
    embedder = SentenceTransformer(str(embedding_model_path))
    embeddings = embedder.encode(
        all_chunks, batch_size=4, show_progress_bar=True
    ).astype("float32")

    # Step 3: Build FAISS index
    print(f"Building FAISS index for {len(all_chunks):,} chunks...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, str(artifacts_dir / f"{index_prefix}.faiss"))
    print(f"FAISS Index built successfully: {index_prefix}.faiss")

    # Step 4: Build BM25 index
    print(f"Building BM25 index for {len(all_chunks):,} chunks...")
    tokenized_chunks = [preprocess_for_bm25(chunk) for chunk in all_chunks]
    bm25_index = BM25Okapi(tokenized_chunks)
    with open(artifacts_dir / f"{index_prefix}_bm25.pkl", "wb") as f:
        pickle.dump(bm25_index, f)
    print(f"BM25 Index built successfully: {index_prefix}_bm25.pkl")

    # Step 5: Dump index artifacts
    with open(artifacts_dir / f"{index_prefix}_chunks.pkl", "wb") as f:
        pickle.dump(all_chunks, f)
    with open(artifacts_dir / f"{index_prefix}_sources.pkl", "wb") as f:
        pickle.dump(sources, f)
    with open(artifacts_dir / f"{index_prefix}_meta.pkl", "wb") as f:
        pickle.dump(metadata, f)
    print(f"Saved all index artifacts with prefix: {index_prefix}")

    # Step 6: Generate and index section summaries (if enabled)
    if generate_summaries:
        print(f"\n{'='*60}")
        print("Building summary index...")
        print(f"{'='*60}\n")
        sections_to_summarize = sections
        summaries = generate_section_summaries(
            sections_to_summarize,
            model_path=summary_model_path,
        )
        print(summaries)
        with open(artifacts_dir / f"{index_prefix}_summaries.pkl", "wb") as f:
            pickle.dump(summaries, f)
        print(f"Saved summaries: {index_prefix}_summaries.pkl")

        # Create embeddings
        summary_texts = [s['summary'] for s in summaries]
        print(f"Embedding {len(summary_texts)} summaries...")
        summary_embeddings = embedder.encode(
            summary_texts, batch_size=4, show_progress_bar=True
        ).astype("float32")
        
        # Build FAISS index for summaries
        summary_index = faiss.IndexFlatL2(dim)
        summary_index.add(summary_embeddings)
        faiss.write_index(summary_index, str(artifacts_dir / f"{index_prefix}_summaries.faiss"))
        print(f"FAISS summary index built: {index_prefix}_summaries.faiss")

    # Step 7: Optional visualization
    if do_visualize:
        visualize(embeddings, sources)


# ------------------------ Helper functions ------------------------------

def preprocess_for_bm25(text: str) -> list[str]:
    """
    Simplifies text to keep only letters, numbers, underscores, hyphens,
    apostrophes, plus, and hash — suitable for BM25 tokenization.
    """
    # Convert to lowercase
    text = text.lower()

    # Keep only allowed characters
    text = re.sub(r"[^a-z0-9_'#+-]", " ", text)

    # Split by whitespace
    tokens = text.split()

    return tokens


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
