"""
Chunking Mode Constants

This module defines constants for chunking strategies to ensure consistent naming
across the codebase.
"""

# Chunking Strategy Names
CHUNK_MODE_CHARS = "chars"
CHUNK_MODE_TOKENS = "tokens" 
CHUNK_MODE_SLIDING_TOKENS = "sliding-tokens"
CHUNK_MODE_SECTIONS = "sections"
CHUNK_MODE_LLM = "llm"
CHUNK_MODE_PROPOSITIONAL = "propositional"

# Available chunking modes for validation
AVAILABLE_CHUNK_MODES = {
    CHUNK_MODE_CHARS,
    CHUNK_MODE_TOKENS,
    CHUNK_MODE_SLIDING_TOKENS,
    CHUNK_MODE_SECTIONS,
    CHUNK_MODE_LLM,
    CHUNK_MODE_PROPOSITIONAL
}

def validate_chunk_mode(chunk_mode: str) -> bool:
    """Validate that the chunk mode is supported."""
    return chunk_mode.lower() in AVAILABLE_CHUNK_MODES
