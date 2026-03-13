"""
LlamaIndex-based RAG pipeline for TokenSmith.

Behaviorally equivalent to the original src/ pipeline:
- Qwen/Qwen3-Embedding-4B via HuggingFace for embeddings
- Qwen2.5-1.5B GGUF model for generation
- Vector + BM25 retrieval with RRF fusion (LlamaIndex built-in modules)
- Cross-encoder reranking (ms-marco-MiniLM-L6-v2)
"""
