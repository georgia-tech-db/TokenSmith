import re
import statistics
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

# -------------------------- Chunking Configs --------------------------

class ChunkConfig(ABC):
    @abstractmethod
    def validate(self):
        pass

    @abstractmethod
    def to_string(self) -> str:
        pass


@dataclass
class SectionRecursiveConfig(ChunkConfig):
    """Configuration for section-based chunking with recursive splitting."""
    recursive_chunk_size: int
    recursive_overlap: int

    def to_string(self) -> str:
        return (
            f"chunk_mode=sections+recursive, "
            f"chunk_size={self.recursive_chunk_size}, "
            f"overlap={self.recursive_overlap}"
        )

    def validate(self):
        assert self.recursive_chunk_size > 0, "recursive_chunk_size must be > 0"
        assert self.recursive_overlap >= 0, "recursive_overlap must be >= 0"
        assert self.recursive_overlap < self.recursive_chunk_size, \
            "recursive_overlap must be less than recursive_chunk_size"


# -------------------------- Chunking Strategies --------------------------

# Separator hierarchy: tries the largest natural boundary first,
# falling back to smaller ones only when a chunk still exceeds chunk_size.
# The "" fallback guarantees a hard character-level split as a last resort.
_RECURSIVE_SEPARATORS = [
    "\n\n",   # paragraph break     — strongest preference
    ". ",     # declarative sentence end
    "? ",     # question end
    "! ",     # exclamation end
]


class ChunkStrategy(ABC):
    """Abstract base for all chunking strategies."""
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def chunk(self, text: str) -> List[str]:
        pass

    @abstractmethod
    def artifact_folder_name(self) -> str:
        pass


class SectionRecursiveStrategy(ChunkStrategy):
    """
    Applies recursive character-based splitting to already-extracted sections.
    Tries paragraph → line → sentence → word → character boundaries in order.
    The "" fallback ensures no chunk can exceed chunk_size_in_chars.
    """

    def __init__(self, config: SectionRecursiveConfig):
        self.config = config
        config.validate()
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.recursive_chunk_size,
            chunk_overlap=config.recursive_overlap,
            separators=_RECURSIVE_SEPARATORS,
            keep_separator=True,
        )

    def name(self) -> str:
        return (
            f"sections+recursive"
            f"({self.config.recursive_chunk_size},{self.config.recursive_overlap})"
        )

    def artifact_folder_name(self) -> str:
        return "sections"

    def chunk(self, text: str) -> List[str]:
        chunks = self._splitter.split_text(text)

        # Post-split validation: the "" separator should prevent this,
        # but log loudly if anything still slips through.
        over_limit = [
            (i, len(c)) for i, c in enumerate(chunks)
            if len(c) > self.config.recursive_chunk_size
        ]
        if over_limit:
            for idx, length in over_limit:
                print(
                    f"[WARNING] Chunk {idx} has {length} chars, "
                    f"exceeding limit of {self.config.recursive_chunk_size}. "
                    f"This should not happen with the '' fallback separator — "
                    f"check for non-standard whitespace or very long tokens."
                )

        # Drop pure-whitespace chunks
        return [c for c in chunks if c.strip()]


# -------------------------- Chunk Stats --------------------------

def print_chunk_stats(chunks: List[str], chunk_size_in_chars: int) -> None:
    """
    Prints a statistical summary of chunk character lengths.
    Useful for diagnosing chunking quality and context window overflow risk.

    Args:
        chunks:             List of chunk strings to analyse.
        chunk_size_in_chars: The configured hard limit, used to flag overflows.
    """
    if not chunks:
        print("[Chunk Stats] No chunks to analyse.")
        return

    lengths = [len(c) for c in chunks]
    total = len(lengths)
    over = [l for l in lengths if l > chunk_size_in_chars]
    pct_over = (len(over) / total) * 100

    # Rough token estimate at 4 chars/token
    est_tokens = [l / 4.0 for l in lengths]

    print("\n" + "="*55)
    print("  CHUNK STATS")
    print("="*55)
    print(f"  Total chunks          : {total:,}")
    print(f"  Char limit            : {chunk_size_in_chars:,}")
    print(f"  --- Character lengths ---")
    print(f"  Min                   : {min(lengths):,}")
    print(f"  Max                   : {max(lengths):,}")
    print(f"  Mean                  : {statistics.mean(lengths):,.1f}")
    print(f"  Median                : {statistics.median(lengths):,.1f}")
    print(f"  Stdev                 : {statistics.stdev(lengths):,.1f}" if total > 1 else "  Stdev                 : N/A")
    print(f"  --- Token estimates (chars ÷ 4) ---")
    print(f"  Min                   : {min(est_tokens):.0f}")
    print(f"  Max                   : {max(est_tokens):.0f}")
    print(f"  Mean                  : {statistics.mean(est_tokens):.1f}")
    print(f"  --- Overflow ---")
    print(f"  Chunks over limit     : {len(over):,} / {total:,}  ({pct_over:.1f}%)")
    if over:
        print(f"  Largest offender      : {max(over):,} chars (~{max(over)/4:.0f} tokens)")

    print(f"  --- Distribution ---")
    buckets = [
        ("  0 – 500  chars", 0, 500),
        ("501 – 1000 chars", 501, 1000),
        ("1001 – 1500 chars", 1001, 1500),
        ("1501 – 2000 chars", 1501, 2000),
        ("2001 – 2500 chars", 2001, 2500),
        ("2500+      chars", 2501, float("inf")),
    ]
    for label, lo, hi in buckets:
        count = sum(1 for l in lengths if lo <= l <= hi)
        bar = "█" * (count * 30 // total) if total else ""
        print(f"  {label}: {count:>5,}  {bar}")
    print("="*55 + "\n")


# ----------------------------- Document Chunker ---------------------------------

class DocumentChunker:
    """
    Chunks text via a provided strategy.
    Table blocks (<table>...</table>) are preserved as atomic units —
    extracted before splitting and restored into whichever chunk holds
    their placeholder.
    """

    TABLE_RE = re.compile(r"<table>.*?</table>", re.DOTALL | re.IGNORECASE)

    def __init__(
        self,
        strategy: Optional[ChunkStrategy],
        keep_tables: bool = True,
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
            chunk = chunk.replace(f"[TABLE_PLACEHOLDER_{i}]", t)
        return chunk

    def _check_split_placeholders(self, chunks: List[str], num_tables: int) -> None:
        """Warn if a table placeholder ended up in more than one chunk."""
        ph_pattern = re.compile(r"\[TABLE_PLACEHOLDER_(\d+)\]")
        seen = {}
        for chunk_idx, chunk in enumerate(chunks):
            for match in ph_pattern.finditer(chunk):
                table_idx = int(match.group(1))
                if table_idx in seen:
                    print(
                        f"[WARNING] TABLE_PLACEHOLDER_{table_idx} appears in "
                        f"both chunk {seen[table_idx]} and chunk {chunk_idx}. "
                        f"Table may be duplicated or lost."
                    )
                seen[table_idx] = chunk_idx

    def chunk(self, text: str) -> List[str]:
        if not text:
            return []

        tables: List[str] = []
        work = text

        if self.keep_tables:
            work, tables = self._extract_tables(work)

        if self.strategy is None:
            raise ValueError("No chunk strategy provided")

        chunks = self.strategy.chunk(work)

        if self.keep_tables and tables:
            self._check_split_placeholders(chunks, len(tables))
            chunks = [self._restore_tables(c, tables) for c in chunks]

        return chunks