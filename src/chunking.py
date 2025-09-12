import re
from abc import ABC, abstractmethod
from typing import List, Optional

from nltk import sent_tokenize
from transformers import AutoTokenizer

# -------------------------- Chunking Strategies --------------------------

class ChunkStrategy(ABC):
    """Abstract base for all chunking strategies."""
    @abstractmethod
    def name(self) -> str: ...
    @abstractmethod
    def chunk(self, text: str) -> List[str]: ...

class CharChunkStrategy(ChunkStrategy):
    def __init__(self, max_chars: int = 20_000):
        self.max_chars = int(max_chars)

    def name(self) -> str:
        return f"chars({self.max_chars})"

    def chunk(self, text: str) -> List[str]:
        step = max(1, self.max_chars)
        return [text[i:i + step] for i in range(0, len(text), step)]

class SentencePackStrategy(ChunkStrategy):
    """
    Sentence-aware packing using word-count as a token proxy.
    """
    def __init__(self, max_tokens: int = 500):
        self.max_tokens = int(max_tokens)

    def name(self) -> str:
        return f"tokens(word-proxy:{self.max_tokens})"

    def chunk(self, text: str) -> List[str]:
        chunks: List[str] = []
        cur, cur_len = [], 0
        for s in sent_tokenize(text):
            w = len(s.split())
            if cur and cur_len + w > self.max_tokens:
                chunks.append(" ".join(cur))
                cur, cur_len = [s], w
            else:
                cur.append(s)
                cur_len += w
        if cur:
            chunks.append(" ".join(cur))
        return chunks

class ParagraphStrategy(ChunkStrategy):
    """
    Splits text into paragraphs based on double newlines.
    Filters out very short paragraphs (likely headings or artifacts).
    """
    def __init__(self, min_chars: int = 50):
        self.min_chars = int(min_chars)

    def name(self) -> str:
        return f"paragraphs(min_chars:{self.min_chars})"

    def chunk(self, text: str) -> List[str]:
        # Split by one or more newlines
        paragraphs = re.split(r'\n\s*\n', text)
        # Filter out empty strings and very short paragraphs.
        return [p.strip() for p in paragraphs if len(p.strip()) >= self.min_chars]

class SlidingTokenStrategy(ChunkStrategy):
    """
    True token windows with overlap using HF tokenizer.
    window = max_tokens, step = max_tokens - overlap_tokens
    """
    def __init__(self,
                 tokenizer_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 max_tokens: int = 350,
                 overlap_tokens: int = 80):
        self.tokenizer_name = tokenizer_name
        self.max_tokens = int(max_tokens)
        self.overlap_tokens = max(0, int(overlap_tokens))
        self._tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True, model_max_length=1_000_000_000)

    def name(self) -> str:
        return f"sliding-tokens({self.max_tokens},{self.overlap_tokens})"

    def chunk(self, text: str) -> List[str]:
        ids = self._tok.encode(text, add_special_tokens=False)
        if not ids or self.max_tokens <= 0:
            return []
        step = self.max_tokens - self.overlap_tokens
        if step <= 0:
            step = self.max_tokens  # safety

        chunks: List[str] = []
        for i in range(0, len(ids), step):
            window = ids[i:i + self.max_tokens]
            if not window:
                break
            chunk_text = self._tok.decode(
                window,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            chunks.append(chunk_text)
        return chunks

# -------------------------- Strategy Factory -----------------------------

def make_chunk_strategy(
    mode: str,
    *,
    chunk_size_char: int,
    chunk_tokens: int,
    tokenizer_name: Optional[str],
) -> ChunkStrategy:
    mode = (mode or "chars").lower()
    if mode == "chars":
        return CharChunkStrategy(max_chars=chunk_size_char)
    if mode == "tokens":
        return SentencePackStrategy(max_tokens=chunk_tokens)
    if mode == "sliding-tokens":
        return SlidingTokenStrategy(
            tokenizer_name=(tokenizer_name or "sentence-transformers/all-MiniLM-L6-v2"),
            max_tokens=chunk_tokens
        )
    if mode == "paragraphs":
        return ParagraphStrategy()
    raise ValueError(f"Unknown chunk_mode: {mode}")
