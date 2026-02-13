"""
LlamaRetriever — evidence-curation RAG pipeline for TokenSmith.

Same retrieval stack as llamaindex (vector + BM25 + RRF + cross-encoder rerank),
but replaces single-shot generation with an iterative agent that:
  1. Selects exact source sentences as evidence
  2. Optionally retrieves more context
  3. Synthesizes a cited answer referencing numbered evidence
"""
