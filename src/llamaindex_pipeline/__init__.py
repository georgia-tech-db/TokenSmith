"""
LlamaIndex-based RAG pipeline for TokenSmith.

A competitive baseline using LlamaIndex with:
- Local QWEN generation model via llama-cpp-python
- HuggingFace sentence-transformer embeddings (<5B)
- Hybrid retrieval (vector + keyword) with cross-encoder reranking
"""
