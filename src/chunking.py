import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter

# -------------------------- Chunking Configs --------------------------
class ChunkConfig(ABC):
    def validate(self):
        pass

    def to_string(self):
        return ""

@dataclass
class SectionRecursiveConfig(ChunkConfig):
    """Configuration for section-based chunking with recursive splitting."""
    recursive_chunk_size: int = 1000
    recursive_overlap: int = 0
    
    def to_string(self):
        return f"chunk_mode=sections+recursive, chunk_size={self.recursive_chunk_size}, overlap={self.recursive_overlap}"

    def validate(self):
        assert self.recursive_chunk_size > 0, "recursive_chunk_size must be > 0"
        assert self.recursive_overlap >= 0, "recursive_overlap must be >= 0"

# -------------------------- Chunking Strategies --------------------------

class ChunkStrategy(ABC):
    """Abstract base for all chunking strategies."""
    @abstractmethod
    def name(self) -> str: ...
    @abstractmethod
    def chunk(self, text: str) -> List[str]: ...
    @abstractmethod
    def artifact_folder_name(self) -> str: ...

class SectionRecursiveStrategy(ChunkStrategy):
    """
    Applies recursive character-based splitting to text.
    This is meant to be used on already-extracted sections.
    """

    def __init__(self, config: SectionRecursiveConfig):
        self.config = config
        self.recursive_chunk_size = config.recursive_chunk_size
        self.recursive_overlap = config.recursive_overlap

    def name(self) -> str:
        return f"sections+recursive({self.recursive_chunk_size},{self.recursive_overlap})"

    def artifact_folder_name(self) -> str:
        return "sections"

    def chunk(self, text: str) -> List[str]:
        """
        Recursively splits text into smaller chunks based on sentence boundaries.
        If a chunk exceeds max_chunk_size, it is further split.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.recursive_chunk_size,
            chunk_overlap=self.recursive_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        return splitter.split_text(text)

# -------------------------- Strategy Factory -----------------------------

def make_chunk_strategy(config: ChunkConfig) -> ChunkStrategy:
    if isinstance(config, SectionRecursiveConfig):
        return SectionRecursiveStrategy(config)
    raise ValueError(f"Unknown chunk config type: {config.__class__.__name__}")

# ----------------------------- Document Chunker ---------------------------------

class DocumentChunker:
    """
    Chunk text via a provided strategy.
    Table blocks (<table>...</table>) are preserved within chunks.
    """

    TABLE_RE = re.compile(r"<table>.*?</table>", re.DOTALL | re.IGNORECASE)

    def __init__(
        self,
        strategy: Optional[ChunkStrategy],
        keep_tables: bool = True
    ):
        self.strategy = strategy
        self.keep_tables = keep_tables

    def _extract_tables(self, text: str) -> Tuple[str, List[str]]:
        tables = self.TABLE_RE.findall(text)
        for i, t in enumerate(tables):
            text = text.replace(t, f"[TABLE_PLACEHOLDER_{i}]")
        return text, tables

    @staticmethod
    def _restore_tables(chunk: str, tables: List[str]) -> str:
        for i, t in enumerate(tables):
            ph = f"[TABLE_PLACEHOLDER_{i}]"
            if ph in chunk:
                chunk = chunk.replace(ph, t)
        return chunk

    def chunk(self, text: str) -> List[str]:
        if not text:
            return []
        work = text
        tables: List[str] = []
        if self.keep_tables:
            work, tables = self._extract_tables(work)

        if self.strategy is None:
            raise ValueError("No chunk strategy provided")
        else:
            chunks = self.strategy.chunk(work)

        if self.keep_tables and tables:
            chunks = [self._restore_tables(c, tables) for c in chunks]
        return chunks
