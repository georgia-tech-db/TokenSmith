import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple

from nltk import sent_tokenize
from transformers import AutoTokenizer


# -------------------------- Chunking Configs --------------------------
class ChunkConfig(ABC):
    def validate(self):
        pass

    def to_string(self):
        return ""

@dataclass
class CharChunkConfig(ChunkConfig):
    max_chars: int

    def to_string(self):
        return f"chunk_mode=chars, max_chars={self.max_chars}"

    def validate(self):
        assert self.max_chars > 0, "chunk_size_char must be > 0"

@dataclass
class TokenChunkConfig(ChunkConfig):
    max_tokens: int

    def to_string(self):
        return f"chunk_mode=tokens, max_tokens={self.max_tokens}"

    def validate(self):
        assert self.max_tokens > 0, "chunk_size_char must be > 0"

@dataclass
class SlidingTokenConfig(ChunkConfig):
    max_tokens: int
    overlap_tokens: int
    tokenizer_name: str

    def to_string(self):
        return (
            f"chunk_mode=sliding-tokens, "
            f"max_tokens={self.max_tokens}, "
            f"overlap_tokens={self.overlap_tokens}, "
            f"tokenizer_name={self.tokenizer_name}"
        )

    def validate(self):
        assert self.max_tokens > 0, "chunk_size_char must be > 0"
        assert self.overlap_tokens > 0, "chunk_size_char must be > 0"

@dataclass
class SectionChunkConfig(ChunkConfig):
    def to_string(self):
        return "chunk_mode=section"

# -------------------------- Chunking Strategies --------------------------

class ChunkStrategy(ABC):
    """Abstract base for all chunking strategies."""
    @abstractmethod
    def name(self) -> str: ...
    @abstractmethod
    def chunk(self, text: str, page_map: List[Tuple[str, int]] = None) -> List[dict]: ...
    @abstractmethod
    def artifact_folder_name(self) -> str: ...

class CharChunkStrategy(ChunkStrategy):
    def __init__(self, config: CharChunkConfig):
        self.max_chars = int(config.max_chars)

    def name(self) -> str:
        return f"chars-{self.max_chars}"

    def artifact_folder_name(self) -> str:
        return f"chars({self.max_chars})"

    def chunk(self, text: str, page_map: List[Tuple[str, int]] = None) -> List[dict]:
        step = max(1, self.max_chars)
        raw_chunks = [text[i:i + step] for i in range(0, len(text), step)]
        return [{"content": chunk} for chunk in raw_chunks]

class SentencePackStrategy(ChunkStrategy):
    """
    Sentence-aware packing using word-count as a token proxy.
    """
    def __init__(self, config: TokenChunkConfig):
        self.max_tokens = int(config.max_tokens)

    def name(self) -> str:
        return f"tokens(word-proxy:{self.max_tokens})"

    def artifact_folder_name(self) -> str:
        return f"tokens-{self.max_tokens}"

    def chunk(self, text: str, page_map: List[Tuple[str, int]] = None) -> List[dict]:
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
        return [{"content": chunk} for chunk in chunks]

class SlidingTokenStrategy(ChunkStrategy):
    """
    True token windows with overlap using HF tokenizer.
    window = max_tokens, step = max_tokens - overlap_tokens
    """
    def __init__(self, config: SlidingTokenConfig):
        self.tokenizer_name = config.tokenizer_name
        self.max_tokens = int(config.max_tokens)
        self.overlap_tokens = int(config.overlap_tokens)
        self._tok = AutoTokenizer.from_pretrained(config.tokenizer_name,
                                                  use_fast=True,
                                                  model_max_length=1_000_000_000)

    def name(self) -> str:
        return f"sliding-tokens({self.max_tokens},{self.overlap_tokens})"

    def artifact_folder_name(self) -> str:
        return f"sliding-tokens-{self.max_tokens}-{self.overlap_tokens}"

    def chunk(self, text: str, page_map: List[Tuple[str, int]] = None) -> List[dict]:
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
        return [{"content": chunk} for chunk in chunks]

class SectionStrategy(ChunkStrategy):
    """
    Splits text into chunks based on numeric section headings.
    Example matches:
      1. Introduction
      2.3 Subtopic
      10.4.1 Deep Dive
    Collects text until the next heading of same-or-higher level.
    """
    HEADING_RE = re.compile(
        r"""
        (?m)                               # multiline
        ^(?=.{,120}$)\s{0,3}               # shortish heading line
        (?P<num>[1-9]\d*(?:\.[0-9]+)*)     # 1.2 or 10.4.1 etc.
        (?![)\]])                          # avoid "1.2)"
        \s+(?P<title>(?!\d).+?)\s*$        # title text
        """,
        re.VERBOSE,
    )

    def name(self) -> str:
        return "sections"

    def artifact_folder_name(self) -> str:
        return "sections"

    def find_page_number(self, pos: int, page_map: List[Tuple[str, int]]) -> int:
        """Find the page number for a given character position."""
        char_count = 0
        for text, page_num in page_map:
            if char_count + len(text) >= pos:
                return page_num
            char_count += len(text)
        return -1 # Fallback

    def chunk(self, text: str, page_map: List[Tuple[str, int]] = None) -> List[dict]:
        matches = list(self.HEADING_RE.finditer(text))
        if not matches:
            # No headings detected → return the whole text as one chunk
            content = text.strip()
            if not content:
                return []
            return [{
                "content": content,
                "page_start": self.find_page_number(0, page_map) if page_map else -1,
                "page_end": self.find_page_number(len(text), page_map) if page_map else -1,
                "section_num": "N/A",
                "section_title": "Preface",
            }]

        heads = []
        for m in matches:
            num = m.group("num")
            title = m.group("title").strip()
            level = num.count(".") + 1
            start_page = self.find_page_number(m.start(), page_map) if page_map else -1
            heads.append({
                "num": num, "title": title, "level": level,
                "start": m.start(), "endline": m.end(),
                "page": start_page
            })

        chunks: List[dict] = []
        N = len(heads)
        for i, h in enumerate(heads):
            end_idx = len(text)
            for j in range(i + 1, N):
                if heads[j]["level"] <= h["level"]:
                    end_idx = heads[j]["start"]
                    break
            body = text[h["endline"]:end_idx].strip("\n").strip()
            
            end_page = self.find_page_number(end_idx, page_map) if page_map else -1
            if h["page"] == end_page:
                page_range = str(h["page"])
            else:
                page_range = f"{h['page']}-{end_page}"

            if body:
                chunks.append({
                    "content": f"{h['title']}\n{body}",
                    "page_start": h["page"],
                    "page_end": end_page,
                    "page_range": page_range,
                    "section_num": h["num"],
                    "section_title": h["title"],
                })

        # If there's preface text before the first heading, keep it as a chunk
        preface_start = 0
        first_start = heads[0]["start"]
        preface = text[preface_start:first_start].strip()
        if preface:
            end_page = self.find_page_number(first_start, page_map) if page_map else -1
            chunks.insert(0, {
                "content": preface,
                "page_start": self.find_page_number(0, page_map) if page_map else -1,
                "page_end": end_page,
                "section_num": "N/A",
                "section_title": "Preface"
            })

        return chunks

# -------------------------- Strategy Factory -----------------------------

def make_chunk_strategy(
    config: ChunkConfig
) -> ChunkStrategy:
    if isinstance(config, CharChunkConfig):
        return CharChunkStrategy(config)
    if isinstance(config, TokenChunkConfig):
        return SentencePackStrategy(config)
    if isinstance(config, SlidingTokenConfig):
        return SlidingTokenStrategy(config)
    if isinstance(config, SectionChunkConfig):
        return SectionStrategy()
    raise ValueError(f"Unknown chunk config type: ", config.__class__.__name__)