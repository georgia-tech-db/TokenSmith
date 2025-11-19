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
import hashlib
from typing import List, Dict

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from src.embedder import SentenceTransformer

from src.preprocessing.chunking import DocumentChunker, ChunkConfig
from src.preprocessing.extraction import extract_sections_from_markdown
from src.config import QueryPlanConfig


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


# ------------------------ Incremental indexing -----------------------------

def build_index_incremental(
    markdown_file: str,
    *,
    chunker: DocumentChunker,
    chunk_config: ChunkConfig,
    embedding_model_path: str,
    artifacts_dir: os.PathLike,
    index_prefix: str,
    do_visualize: bool = False,
    indexing_config: Dict = None,
) -> None:
    """
    Build index incrementally, only re-embedding changed sections.
    Uses content hashing to detect changes.
    """
    artifacts_dir = pathlib.Path(artifacts_dir)
    cache_dir = artifacts_dir
    if indexing_config and indexing_config.get("cache_dir"):
        cache_dir = pathlib.Path(indexing_config["cache_dir"])
        cache_dir.mkdir(parents=True, exist_ok=True)
    
    cache_file = cache_dir / f"{index_prefix}_section_cache.pkl"
    
    # Load existing cache
    section_cache = {}
    if cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                section_cache = pickle.load(f)
            print(f"Loaded cache with {len(section_cache)} sections")
        except Exception as e:
            print(f"Warning: Could not load cache: {e}. Starting fresh.")
            section_cache = {}
    
    # Extract sections
    sections = extract_sections_from_markdown(
        markdown_file,
        exclusion_keywords=DEFAULT_EXCLUSION_KEYWORDS
    )
    
    all_chunks: List[str] = []
    all_embeddings_list: List[List[float]] = []
    sources: List[str] = []
    metadata: List[Dict] = []
    page_to_chunk_ids = {}
    current_page = 1
    total_chunks = 0
    
    to_embed_sections = []
    to_embed_indices = []
    cached_count = 0
    
    # Process each section
    for section_idx, c in enumerate(sections):
        # Create content hash (heading + content)
        content_hash = hashlib.md5((c['heading'] + c['content']).encode()).hexdigest()
        
        if content_hash in section_cache:
            # Use cached data
            cached = section_cache[content_hash]
            section_chunks = cached['chunks']
            section_embeddings = cached['embeddings']
            section_metadata = cached['metadata']
            section_page_map = cached.get('page_to_chunk_ids', {})
            
            # Update global structures
            start_chunk_id = total_chunks
            for i, chunk in enumerate(section_chunks):
                all_chunks.append(chunk)
                all_embeddings_list.append(section_embeddings[i])
                sources.append(markdown_file)
                # Update chunk_id in metadata
                chunk_meta = section_metadata[i].copy()
                chunk_meta["chunk_id"] = start_chunk_id + i
                metadata.append(chunk_meta)
                
                # Update page mapping - section_page_map maps page -> [chunk indices within section]
                for page, chunk_indices in section_page_map.items():
                    if i in chunk_indices:
                        page_to_chunk_ids.setdefault(page, set()).add(start_chunk_id + i)
            
            # Update current_page to last page in this section
            if section_page_map:
                current_page = max(section_page_map.keys())
            
            total_chunks += len(section_chunks)
            cached_count += 1
            print(f"[CACHED] Using cached: {c['heading'][:50]}... ({len(section_chunks)} chunks)")
        else:
            # Mark for embedding
            to_embed_sections.append((section_idx, c, content_hash))
            print(f"⚡ Will embed: {c['heading'][:50]}...")
    
    # Embed new/changed sections
    if to_embed_sections:
        print(f"\nEmbedding {len(to_embed_sections)} new/changed sections...")
        
        # Process sections to get chunks and metadata (same as regular build_index)
        new_chunks_list = []
        new_metadata_list = []
        new_page_maps = []
        
        for section_idx, c, content_hash in to_embed_sections:
            has_table = bool(TABLE_RE.search(c['content']))
            chapter_num = extract_chapter_number(c['heading'])
            section_hierarchy = parse_section_hierarchy(c['heading'])
            section_type = classify_section_type(c['heading'], c['content'])
            keywords = extract_keywords(c['content'], top_n=10)
            
            meta_template = {
                "filename": markdown_file,
                "mode": chunk_config.to_string(),
                "keep_tables": chunker.keep_tables,
                "has_table": has_table,
                "section": c['heading'],
                "chapter": chapter_num,
                "section_hierarchy": section_hierarchy,
                "section_type": section_type,
                "keywords": keywords
            }
            
            sub_chunks = chunker.chunk(c['content'])
            page_pattern = re.compile(r'--- Page (\d+) ---')
            section_page_map = {}
            section_metadata = []
            section_current_page = current_page  # Start with current page from previous section
            
            for sub_chunk_id, sub_chunk in enumerate(sub_chunks):
                # Track page for this chunk
                chunk_start_page = section_current_page
                fragments = page_pattern.split(sub_chunk)
                for i in range(1, len(fragments), 2):
                    try:
                        page_num = int(fragments[i])
                        section_current_page = page_num
                        section_page_map.setdefault(page_num, []).append(sub_chunk_id)
                    except (IndexError, ValueError):
                        continue
                
                sub_chunk = re.sub(page_pattern, '', sub_chunk).strip()
                sub_chunk_meta = meta_template.copy()
                sub_chunk_meta["page_number"] = section_current_page
                sub_chunk_meta["chunk_position_in_section"] = sub_chunk_id
                sub_chunk_meta["total_chunks_in_section"] = len(sub_chunks)
                
                new_chunks_list.append(sub_chunk)
                section_metadata.append(sub_chunk_meta)
            
            new_metadata_list.append(section_metadata)
            new_page_maps.append(section_page_map)
            to_embed_indices.append((content_hash, len(sub_chunks)))
            
            # Update current_page for next section
            if section_page_map:
                current_page = max(section_page_map.keys())
        
        # Embed all new chunks
        if new_chunks_list:
            embedder = SentenceTransformer(embedding_model_path)
            use_parallel = indexing_config and indexing_config.get("use_parallel_embedding", False)
            batch_size = indexing_config.get("batch_size", 32) if indexing_config else 32
            
            import time
            start_time = time.time()
            
            if use_parallel and len(new_chunks_list) >= 100:
                new_embeddings = embedder.encode_parallel(
                    new_chunks_list,
                    batch_size=batch_size,
                    show_progress_bar=True
                )
            else:
                new_embeddings = embedder.encode(
                    new_chunks_list,
                    batch_size=batch_size,
                    show_progress_bar=True
                )
            
            elapsed = time.time() - start_time
            print(f"Embedding completed in {elapsed:.2f} seconds ({len(new_chunks_list)/elapsed:.1f} chunks/sec)")
            
            # Update cache and merge with existing data
            chunk_offset = 0
            for (content_hash, num_chunks), section_metadata, section_page_map in zip(
                to_embed_indices, new_metadata_list, new_page_maps
            ):
                section_chunks = new_chunks_list[chunk_offset:chunk_offset+num_chunks]
                section_embeddings = new_embeddings[chunk_offset:chunk_offset+num_chunks].tolist()
                
                # Store in cache
                section_cache[content_hash] = {
                    'chunks': section_chunks,
                    'embeddings': section_embeddings,
                    'metadata': section_metadata,
                    'page_to_chunk_ids': section_page_map
                }
                
                # Add to global structures
                start_chunk_id = total_chunks
                for i, chunk in enumerate(section_chunks):
                    all_chunks.append(chunk)
                    all_embeddings_list.append(section_embeddings[i])
                    sources.append(markdown_file)
                    metadata.append(section_metadata[i])
                    
                    # Update page mapping
                    for page, chunk_ids in section_page_map.items():
                        if i in chunk_ids:
                            page_to_chunk_ids.setdefault(page, set()).add(start_chunk_id + i)
                
                total_chunks += num_chunks
                chunk_offset += num_chunks
            
            # Save updated cache
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(section_cache, f)
                print(f"Saved updated cache with {len(section_cache)} sections")
            except Exception as e:
                print(f"Warning: Could not save cache: {e}")
    
    print(f"\nCache statistics: {cached_count} sections reused, {len(to_embed_sections)} sections embedded")
    
    # Convert embeddings to numpy array
    embeddings = np.array(all_embeddings_list, dtype=np.float32)
    
    # Convert page mapping
    final_map = {}
    for page, id_set in page_to_chunk_ids.items():
        final_map[page] = sorted(list(id_set))
    
    output_file = artifacts_dir / f"{index_prefix}_page_to_chunk_map.json"
    with open(output_file, "w") as f:
        json.dump(final_map, f, indent=2)
    print(f"Saved page to chunk ID map: {output_file}")
    
    # Build FAISS index
    print(f"Building FAISS index for {len(all_chunks):,} chunks...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, str(artifacts_dir / f"{index_prefix}.faiss"))
    print(f"FAISS Index built successfully: {index_prefix}.faiss")
    
    # Build BM25 index
    print(f"Building BM25 index for {len(all_chunks):,} chunks...")
    tokenized_chunks = [preprocess_for_bm25(chunk) for chunk in all_chunks]
    bm25_index = BM25Okapi(tokenized_chunks)
    with open(artifacts_dir / f"{index_prefix}_bm25.pkl", "wb") as f:
        pickle.dump(bm25_index, f)
    print(f"BM25 Index built successfully: {index_prefix}_bm25.pkl")
    
    # Dump index artifacts
    with open(artifacts_dir / f"{index_prefix}_chunks.pkl", "wb") as f:
        pickle.dump(all_chunks, f)
    with open(artifacts_dir / f"{index_prefix}_sources.pkl", "wb") as f:
        pickle.dump(sources, f)
    with open(artifacts_dir / f"{index_prefix}_meta.pkl", "wb") as f:
        pickle.dump(metadata, f)
    print(f"Saved all index artifacts with prefix: {index_prefix}")
    
    if do_visualize:
        visualize(embeddings, sources)


# ------------------------ Metadata helper functions -----------------------------

def extract_chapter_number(heading: str) -> int:
    """Extract chapter number from heading like '## 1.1 Database-System Applications'"""
    match = re.search(r'##\s+(\d+)', heading)
    return int(match.group(1)) if match else 0


def parse_section_hierarchy(heading: str) -> Dict:
    """Parse heading into chapter.section.subsection structure"""
    # Extract pattern like "1.2.3" or "1.1"
    match = re.search(r'##\s+(\d+(?:\.\d+)*)', heading)
    if match:
        parts = match.group(1).split('.')
        return {
            'chapter': int(parts[0]) if len(parts) > 0 else 0,
            'section': int(parts[1]) if len(parts) > 1 else 0,
            'subsection': int(parts[2]) if len(parts) > 2 else 0,
        }
    return {'chapter': 0, 'section': 0, 'subsection': 0}


def classify_section_type(heading: str, content: str) -> str:
    """Classify section as introduction, definition, example, algorithm, etc."""
    heading_lower = heading.lower()
    content_lower = content.lower()[:500]  # Check first 500 chars
    
    if 'introduction' in heading_lower or 'overview' in heading_lower:
        return 'introduction'
    elif 'example' in heading_lower or 'examples' in heading_lower:
        return 'example'
    elif 'algorithm' in heading_lower or 'procedure' in heading_lower or 'steps' in heading_lower:
        return 'algorithm'
    elif 'summary' in heading_lower or 'conclusion' in heading_lower:
        return 'summary'
    elif 'definition' in heading_lower or 'define' in content_lower[:200]:
        return 'definition'
    elif 'exercise' in heading_lower or 'problem' in heading_lower:
        return 'exercise'
    else:
        return 'content'


def extract_keywords(content: str, top_n: int = 10) -> List[str]:
    """Extract top N keywords from content using simple frequency analysis."""
    # Simple word frequency (excluding stopwords)
    stopwords = set([
        'the', 'is', 'at', 'which', 'on', 'for', 'a', 'an', 'and', 'or', 'in',
        'to', 'of', 'by', 'with', 'that', 'this', 'it', 'as', 'are', 'was',
        'what', 'when', 'where', 'who', 'how', 'be', 'been', 'have', 'has',
        'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may',
        'might', 'must', 'can', 'from', 'up', 'about', 'into', 'through', 'during'
    ])
    
    # Extract words (alphanumeric sequences)
    words = re.findall(r'\b[a-zA-Z]{3,}\b', content.lower())
    
    # Count frequencies
    word_freq = {}
    for word in words:
        if word not in stopwords:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Sort by frequency and return top N
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, _ in sorted_words[:top_n]]

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
    indexing_config: Dict = None,
) -> None:
    """
    Extract sections, chunk, embed, and build both FAISS and BM25 indexes.

    Persists:
        - {prefix}.faiss
        - {prefix}_bm25.pkl
        - {prefix}_chunks.pkl
        - {prefix}_sources.pkl
        - {prefix}_meta.pkl
    
    If use_incremental is enabled in indexing_config, uses incremental indexing.
    """
    # Check if incremental indexing is enabled
    use_incremental = False
    if indexing_config:
        use_incremental = indexing_config.get("use_incremental", False)
    
    if use_incremental:
        print("Using incremental indexing mode...")
        return build_index_incremental(
            markdown_file=markdown_file,
            chunker=chunker,
            chunk_config=chunk_config,
            embedding_model_path=embedding_model_path,
            artifacts_dir=artifacts_dir,
            index_prefix=index_prefix,
            do_visualize=do_visualize,
            indexing_config=indexing_config,
        )
    
    # Original sequential indexing logic continues below...
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
        
        # Extract enhanced metadata
        chapter_num = extract_chapter_number(c['heading'])
        section_hierarchy = parse_section_hierarchy(c['heading'])
        section_type = classify_section_type(c['heading'], c['content'])
        keywords = extract_keywords(c['content'], top_n=10)
        
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
            "page_number": None,
            # Enhanced metadata fields
            "chapter": chapter_num,
            "section_hierarchy": section_hierarchy,
            "section_type": section_type,
            "keywords": keywords
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
            
            # Create metadata for this specific sub-chunk
            sub_chunk_meta = meta.copy()
            sub_chunk_meta["page_number"] = current_page
            sub_chunk_meta["chunk_id"] = total_chunks + sub_chunk_id
            sub_chunk_meta["chunk_position_in_section"] = sub_chunk_id
            sub_chunk_meta["total_chunks_in_section"] = len(sub_chunks)
            metadata.append(sub_chunk_meta)

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
    
    # Get indexing config
    use_parallel = False
    num_workers = None
    batch_size = 32  # Increased default
    if indexing_config:
        use_parallel = indexing_config.get("use_parallel_embedding", False)
        num_workers = indexing_config.get("num_workers")
        batch_size = indexing_config.get("batch_size", 32)
    
    embedder = SentenceTransformer(embedding_model_path)
    
    import time
    start_time = time.time()
    
    if use_parallel and len(all_chunks) >= 100:
        # Use parallel encoding for large batches
        embeddings = embedder.encode_parallel(
            all_chunks,
            num_workers=num_workers,
            batch_size=batch_size,
            show_progress_bar=True
        )
    else:
        # Use regular encoding
        embeddings = embedder.encode(
            all_chunks, batch_size=batch_size, show_progress_bar=True
        )
    
    elapsed = time.time() - start_time
    print(f"Embedding completed in {elapsed:.2f} seconds ({len(all_chunks)/elapsed:.1f} chunks/sec)")

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

    # # Step 6: Optional visualization
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
