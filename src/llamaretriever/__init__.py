"""
LlamaRetriever — BookRAG-style hierarchical RAG pipeline for TokenSmith.

Indexes documents as a section tree + entity graph, retrieves at section
level first, then narrows to leaf passages within selected subtrees.
Pipeline: classify → section-select → leaf-retrieve → synthesize.
"""
