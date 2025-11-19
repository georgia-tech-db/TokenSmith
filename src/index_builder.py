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
import json
import numpy as np
import torch
from typing import List, Dict

import faiss
from rank_bm25 import BM25Okapi, BM25Plus, BM25L
from sentence_transformers import SentenceTransformer
import yaml

from src.preprocessing.chunking import DocumentChunker, ChunkConfig
from src.preprocessing.extraction import extract_sections_from_markdown
from src.config import QueryPlanConfig

from multiprocessing import Pool, cpu_count

# use GPU instead of CPU to try and fasten the index building process
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


# ----- runtime parallelism knobs (avoid oversubscription) -----
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

TABLE_RE = re.compile(r"<table>.*?</table>", re.DOTALL | re.IGNORECASE)

# Default keywords to exclude sections
DEFAULT_EXCLUSION_KEYWORDS = ['questions', 'exercises', 'summary', 'references']

# ------------------------ Main index builder -----------------------------

def build_index(
    markdown_file: str,
    *,
    chunker: DocumentChunker,
    chunk_config: ChunkConfig,
    embedding_model_path: str,
    artifacts_dir: os.PathLike,
    index_prefix: str, 
    do_visualize: bool = False,
    do_calculate_score: bool = False,
    bm25_type: str,
) -> None:
    """
    Extract sections, chunk, embed, and build both FAISS and BM25 indexes.

    Persists:
        - {prefix}.faiss
        - {prefix}_bm25.pkl
        - {prefix}_chunks.pkl
        - {prefix}_sources.pkl
        - {prefix}_meta.pkl
    """
    all_chunks: List[str] = []
    sources: List[str] = []
    metadata: List[Dict] = []

    # Extract sections from markdown. Exclude some with certain
    # keywords if required.
    sections = extract_sections_from_markdown(
        markdown_file,
        exclusion_keywords=DEFAULT_EXCLUSION_KEYWORDS
    )

    page_to_chunk_ids = {}
    current_page = 1
    total_chunks = 0

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
            "text_preview": c['content'][:100],
            "page_number": None
        }
        
        # Use DocumentChunker to recursively split this section
        sub_chunks = chunker.chunk(c['content'])

        # Regex to find page markers like "--- Page 3 ---"
        page_pattern = re.compile(r'--- Page (\d+) ---')

        # Iterate through each chunk with its index (chunk_id)
        for sub_chunk_id, sub_chunk in enumerate(sub_chunks):
            
            # 1. This sub_chunk starts on the 'current_page'.
            #    Add this sub_chunk_id to the set for the current page.
            page_to_chunk_ids.setdefault(current_page, set()).add(total_chunks+sub_chunk_id)

            # 2. Split the sub_chunk by page markers to see if it
            #    spans multiple pages.
            fragments = page_pattern.split(sub_chunk)
            
            # 3. Process the new pages found within this sub_chunk
            #    We step by 2: (index 1, 2), (index 3, 4), ...
            for i in range(1, len(fragments), 2):
                try:
                    # The first item in the pair is the page number string
                    page_num_str = fragments[i]
                    
                    # Update our "current page" state
                    current_page = int(page_num_str)
                    
                    # The text *after* this marker (at fragments[i+1])
                    # also belongs to this sub_chunk_id. So, add this
                    # sub_chunk_id to the set for the *new* current page.
                    page_to_chunk_ids.setdefault(current_page, set()).add(total_chunks+sub_chunk_id)

                except (IndexError, ValueError):
                    continue

            # Clean sub_chunk by removing page markers
            sub_chunk = re.sub(page_pattern, '', sub_chunk).strip()

            all_chunks.append(sub_chunk)
            sources.append(markdown_file)
            meta["page_number"] = current_page
            metadata.append(meta)

        current_page = next(reversed(page_to_chunk_ids))
        total_chunks += len(sub_chunks)

    # Convert the sets to sorted lists for a clean, predictable output
    final_map = {}
    for page, id_set in page_to_chunk_ids.items():
        final_map[page] = sorted(list(id_set))
    

    output_file = artifacts_dir / f"{index_prefix}_page_to_chunk_map.json"
    with open(output_file, "w") as f:
        json.dump(final_map, f, indent=2)
    print(f"Saved page to chunk ID map: {output_file}")

        
    # Step 2: Create embeddings for FAISS index
    print(f"Embedding {len(all_chunks):,} chunks with {pathlib.Path(embedding_model_path).stem} ...")
    embedder = SentenceTransformer(embedding_model_path, device=device)
    embeddings = embedder.encode(
        all_chunks, batch_size=4, show_progress_bar=True
    )

    # Step 3: Build FAISS index
    print(f"Building FAISS index for {len(all_chunks):,} chunks...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, str(artifacts_dir / f"{index_prefix}.faiss"))
    print(f"FAISS Index built successfully: {index_prefix}.faiss")

    # Step 4: Build BM25 index
    print(f"Building BM25 index for {len(all_chunks):,} chunks...")
    # with Pool(cpu_count()) as p:
    #     tokenized_chunks = p.map(preprocess_for_bm25, all_chunks)
    tokenized_chunks = [preprocess_for_bm25(chunk) for chunk in all_chunks]
    # use different indexing strategy https://www.cs.otago.ac.nz/homepages/andrew/papers/2014-2.pdf
    if bm25_type == 'bm25Plus':
        print('bm25Plus')
        bm25_index = BM25Plus(tokenized_chunks)
    elif bm25_type == 'bm25L':
        print('bm25L')
        bm25_index = BM25L(tokenized_chunks)
    else:
        print('bm25')
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

    # # Step 6: Optional visualization
    if do_visualize:
        visualize(embeddings, sources)

    if do_calculate_score:
        # calculate scores
        evaluate_existing_indexes(artifacts_dir, bm25_type)

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

def stronger_preprocess_for_bm25(text: str) -> list[str]:
    text = text.lower()

    # allow more characters that might be relevant for textbook notations (like <, >, =,...)
    text = re.sub(r"[^a-z0-9_'#+./*^<>=-]", " ", text)

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

# due to the fact that I cannot run the tests due to not having a Google API key, I am writing the methods here. Normally
# it would make somewhat more sense to test this on the test folder

def evaluate_bm25_index(
    bm25_index,
    example_queries: List[str],
    golden_indices_list: List[List[int]],
    k: int = 10
) -> Dict[str, float]:
    
    precisions = []
    recalls = []
    mean_reciprocal_ranks = []
    
    for query, golden_indices in zip(example_queries, golden_indices_list):

        # get bm25 scores for the data
        tokenized_query = preprocess_for_bm25(query)
        scores = bm25_index.get_scores(tokenized_query)
        
        # get top k indices based on the score
        sorted_indices = np.argsort(scores)
        top_k_indices = sorted_indices[-k:][::-1]

        # get how many golden indices were retrived
        golden_indices_retrieved = len(set(top_k_indices).intersection(set(golden_indices)))
        

        # calculate precision and recall
        precision = 0
        if k > 0:
            precision = golden_indices_retrieved / k 

        recall = 1.0
        if len(golden_indices) != 0:
            recall = golden_indices_retrieved / len(set(golden_indices))
        
        # calculate reciprocal_rank score
        mean_reciprocal_rank_score = 0
        # start rank = 1 instead of 0-based to avoid division by 0
        for rank, index in enumerate(top_k_indices, 1):
            # if we find our golden index, calculate MRR and break
            if index in golden_indices:
                mean_reciprocal_rank_score = 1 / rank
                break
        
        precisions.append(precision)
        recalls.append(recall)
        mean_reciprocal_ranks.append(mean_reciprocal_rank_score)
    
    # get precision@k, recall@k, mean reciprocal rank, f1 score
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    avg_mrr = np.mean(mean_reciprocal_ranks)
    f1_score = 0
    if (avg_precision + avg_recall) > 0:
        f1_score = 2 * avg_precision * avg_recall / (avg_precision + avg_recall) 
    
    return {
        'precision': avg_precision,
        'recall': avg_recall,
        'mrr': avg_mrr,
        'f1': f1_score
    }
    
def evaluate_existing_indexes(
    artifacts_dir: pathlib.Path,
    bm25_type: str,
    k: int = 10
) -> Dict[str, float]:

    # get bm25 and chunks path
    bm25_path = artifacts_dir / f"textbook_index_bm25.pkl"
    chunks_path = artifacts_dir / f"textbook_index_chunks.pkl"
    
    
    # load the files
    with open(bm25_path, "rb") as f:
        bm25_index = pickle.load(f)
    
    with open(chunks_path, "rb") as f:
        all_chunks = pickle.load(f)

    # build the benchmark index to get example labels
    example_queries, golden_indices_list = build_benchmark_index(all_chunks)
    bm25_results = evaluate_bm25_index(
        bm25_index=bm25_index,
        example_queries=example_queries,
        golden_indices_list=golden_indices_list,
        k=k
    )

    # print the result
    print(f"result for {bm25_type} is {bm25_results}")
    
    return bm25_results

def build_benchmark_index(all_chunks: List[str]) -> tuple[List[str], List[List[int]]]:
    benchmark_file = "tests/benchmarks.yaml"
    with open(benchmark_file) as f:
        data = yaml.safe_load(f)

    # load the benchmark questions, answers, keywords for golden chunks
    all_benchmarks = data["benchmarks"]
    benchmark_list = []
    for bench in all_benchmarks:
        benchmark_map = {
            "question": bench["question"],
            "answer": bench["expected_answer"],
            "keywords": bench["keywords"],
        }

        benchmark_list.append(benchmark_map)

    example_queries = []
    golden_indices_list = []

    for benchmark in benchmark_list:
        question = benchmark["question"]
        keywords = benchmark["keywords"]
        
        example_queries.append(question)
        golden_chunks = []
        # check if keyword appears in chunk. If it does, the chunk is considered golden
        for idx, chunk in enumerate(all_chunks):
            chunk_lower = chunk.lower()
            
            chunk_words = set(chunk_lower.split())
            keyword_matches = 0
            for k in keywords:
                if k.lower() in chunk_words:
                    keyword_matches += 1
            
            # if there are one matching keyword, we treat it as a golden chunk
            if keyword_matches >= 1:
                golden_chunks.append(idx)
        
        golden_indices_list.append(golden_chunks)
    
    return example_queries, golden_indices_list