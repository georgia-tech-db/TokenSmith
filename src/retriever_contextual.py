"""
Contextual retrieval with hierarchical expansion and cross-reference boosting.

This module provides:
- ContextualRetriever: Expands retrieval with neighbor chunks from same section
- CrossReferenceBooster: Boosts chunks based on textbook index cross-references
"""

from __future__ import annotations

import json
import pathlib
from collections import defaultdict
from typing import Dict, List, Optional

from src.retriever import Retriever


class ContextualRetriever:
    """
    Retrieves chunks with hierarchical context expansion.
    Uses metadata to find neighboring chunks and applies decay scoring.
    """
    
    def __init__(self, base_retrievers: List[Retriever], metadata: List[Dict], 
                 expansion_window: int = 2, decay_factor: float = 0.5):
        """
        Args:
            base_retrievers: List of FAISSRetriever, BM25Retriever, etc.
            metadata: Chunk metadata with section info and page numbers
            expansion_window: How many neighbor chunks to include (each side)
            decay_factor: Score multiplier for neighbor chunks
        """
        self.base_retrievers = base_retrievers
        self.metadata = metadata
        self.expansion_window = expansion_window
        self.decay_factor = decay_factor
        
        # Build section-to-chunks mapping
        self.section_map = self._build_section_map()
        self.chapter_map = self._build_chapter_map()
    
    def _build_section_map(self) -> Dict[str, List[int]]:
        """Group chunk IDs by section heading."""
        section_map = defaultdict(list)
        for idx, meta in enumerate(self.metadata):
            section = meta.get('section', 'unknown')
            section_map[section].append(idx)
        return dict(section_map)
    
    def _build_chapter_map(self) -> Dict[int, List[int]]:
        """Group chunk IDs by chapter number."""
        chapter_map = defaultdict(list)
        for idx, meta in enumerate(self.metadata):
            chapter = meta.get('chapter', 0)
            if chapter > 0:
                chapter_map[chapter].append(idx)
        return dict(chapter_map)
    
    def get_scores_with_context(self, query: str, pool_size: int, 
                                chunks: List[str]) -> Dict[int, float]:
        """
        Retrieve chunks with context expansion.
        
        Returns:
            Dict mapping chunk_id → contextual_score
        """
        # Step 1: Get base scores from FAISS/BM25
        base_scores = {}
        for retriever in self.base_retrievers:
            retriever_scores = retriever.get_scores(query, pool_size, chunks)
            # Merge scores (take max if chunk appears in multiple retrievers)
            for idx, score in retriever_scores.items():
                base_scores[idx] = max(base_scores.get(idx, 0), score)
        
        # Step 2: Expand with neighboring chunks
        expanded_scores = self._expand_within_section(base_scores, chunks)
        
        return expanded_scores
    
    def _expand_within_section(self, base_scores: Dict[int, float], 
                               chunks: List[str]) -> Dict[int, float]:
        """
        For each high-scoring chunk, add its neighbors with decayed scores.
        """
        expanded = dict(base_scores)  # Start with base scores
        
        for chunk_id, score in base_scores.items():
            if chunk_id >= len(self.metadata):
                continue
                
            section = self.metadata[chunk_id].get('section')
            if not section or section not in self.section_map:
                continue
            
            # Find position in section
            section_chunks = self.section_map[section]
            try:
                pos = section_chunks.index(chunk_id)
            except ValueError:
                continue
            
            # Add neighbors with decay
            for offset in range(1, self.expansion_window + 1):
                # Previous neighbors
                if pos - offset >= 0:
                    neighbor_id = section_chunks[pos - offset]
                    neighbor_score = score * (self.decay_factor ** offset)
                    expanded[neighbor_id] = max(
                        expanded.get(neighbor_id, 0), 
                        neighbor_score
                    )
                
                # Next neighbors
                if pos + offset < len(section_chunks):
                    neighbor_id = section_chunks[pos + offset]
                    neighbor_score = score * (self.decay_factor ** offset)
                    expanded[neighbor_id] = max(
                        expanded.get(neighbor_id, 0),
                        neighbor_score
                    )
        
        return expanded
    
    def _is_complex_query(self, query: str) -> bool:
        """Detect multi-concept queries: 'compare', 'difference', 'vs', etc."""
        keywords = ['compare', 'difference', 'versus', 'vs', 'relationship', 
                   'how does', 'why does', 'explain', 'contrast']
        return any(kw in query.lower() for kw in keywords)


class CrossReferenceBooster:
    """
    Boosts chunk scores based on textbook index cross-references.
    If a query mentions "ACID properties", boost chunks from pages listed in the index.
    """
    
    def __init__(self, index_path: str, page_to_chunk_path: str, 
                 boost_factor: float = 1.3):
        """
        Args:
            index_path: Path to extracted_index.json
            page_to_chunk_path: Path to page_to_chunk_map.json
            boost_factor: Multiplier for cross-referenced chunks
        """
        self.boost_factor = boost_factor
        
        # Load textbook index
        try:
            with open(index_path, 'r') as f:
                self.concept_to_pages = json.load(f)
        except FileNotFoundError:
            print(f"Warning: Index file not found at {index_path}. Cross-reference boosting disabled.")
            self.concept_to_pages = {}
        
        # Load page-to-chunk mapping
        try:
            with open(page_to_chunk_path, 'r') as f:
                self.page_to_chunks = json.load(f)
        except FileNotFoundError:
            print(f"Warning: Page-to-chunk map not found at {page_to_chunk_path}. Cross-reference boosting disabled.")
            self.page_to_chunks = {}
        
        # Build reverse index: term → chunk_ids
        self.term_to_chunks = self._build_term_index()
    
    def _build_term_index(self) -> Dict[str, set]:
        """Map each index term to chunk IDs."""
        term_index = {}
        for term, pages in self.concept_to_pages.items():
            chunk_ids = set()
            for page in pages:
                chunks = self.page_to_chunks.get(str(page), [])
                chunk_ids.update(chunks)
            term_index[term.lower()] = chunk_ids
        return term_index
    
    def boost_scores(self, query: str, scores: Dict[int, float]) -> Dict[int, float]:
        """
        Boost scores for chunks that match query terms in the index.
        """
        # Extract potential concepts from query
        query_terms = self._extract_concepts(query)
        
        # Find all chunks related to these concepts
        relevant_chunks = set()
        for term in query_terms:
            if term in self.term_to_chunks:
                relevant_chunks.update(self.term_to_chunks[term])
        
        # Boost scores
        boosted = dict(scores)
        for chunk_id in relevant_chunks:
            if chunk_id in boosted:
                boosted[chunk_id] *= self.boost_factor
        
        return boosted
    
    def _extract_concepts(self, query: str) -> List[str]:
        """
        Extract potential database concepts from query.
        Uses fuzzy matching against index terms.
        """
        query_lower = query.lower()
        concepts = []
        
        # Direct substring matching
        for term in self.term_to_chunks.keys():
            # Check if term appears in query or query words appear in term
            query_words = set(query_lower.split())
            term_words = set(term.split())
            
            if term in query_lower or any(word in term for word in query_words if len(word) > 3):
                concepts.append(term)
        
        return concepts

