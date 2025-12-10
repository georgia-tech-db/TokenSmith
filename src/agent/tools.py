"""
Agent tools for dynamic context retrieval.

Tools:
- IndexScout: Semantic search returning metadata (chunk IDs, relevance)
- NavigationalReader: Read chunk slices with relative offsets
- GrepSearch: Regex search across raw markdown
- SectionSummarizer: Get section content by name
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import faiss
import numpy as np

from src.embedder import SentenceTransformer
from src.retriever import _get_embedder


@dataclass
class ChunkMetadata:
    chunk_id: int
    score: float
    source: str
    preview: str


@dataclass
class GrepMatch:
    line_number: int
    content: str
    context_before: List[str]
    context_after: List[str]


class IndexScout:
    """Semantic search that returns structured metadata."""

    def __init__(
        self,
        faiss_index: faiss.Index,
        chunks: List[str],
        sources: List[str],
        embed_model: str,
    ):
        self.faiss_index = faiss_index
        self.chunks = chunks
        self.sources = sources
        self.embedder = _get_embedder(embed_model)

    def search_index(self, query: str, top_k: int = 10) -> List[ChunkMetadata]:
        """Search index and return metadata list without full chunk text."""
        q_vec = self.embedder.encode([query]).astype("float32")

        if q_vec.shape[1] != self.faiss_index.d:
            raise ValueError(
                f"Embedding dim mismatch: index={self.faiss_index.d} vs query={q_vec.shape[1]}"
            )

        distances, indices = self.faiss_index.search(q_vec, top_k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            score = 1.0 / (1.0 + float(dist))
            preview = self.chunks[idx][:150].replace("\n", " ")
            results.append(
                ChunkMetadata(
                    chunk_id=int(idx),
                    score=score,
                    source=self.sources[idx] if idx < len(self.sources) else "unknown",
                    preview=preview,
                )
            )
        return results

    def format_result(self, results: List[ChunkMetadata]) -> str:
        """Format as machine-readable structured output."""
        if not results:
            return "No results found."
        lines = ["Search results (use chunk_id for read_content):"]
        for i, r in enumerate(results):
            lines.append(f"  [{i}] chunk_id={r.chunk_id} score={r.score:.3f} source={r.source}")
            lines.append(f"      preview: {r.preview}")
        return "\n".join(lines)


class NavigationalReader:
    """Read chunks with relative offset navigation."""

    def __init__(self, chunks: List[str], sources: List[str]):
        self.chunks = chunks
        self.sources = sources

    def read_content(
        self,
        target_chunk_id: int,
        relative_start: int = 0,
        relative_end: int = 0,
    ) -> Tuple[str, List[int]]:
        """
        Fetch chunks[target + start : target + end + 1].
        Returns (concatenated_text, list_of_chunk_ids).
        """
        start_idx = max(0, target_chunk_id + relative_start)
        end_idx = min(len(self.chunks), target_chunk_id + relative_end + 1)

        if start_idx >= len(self.chunks) or end_idx <= start_idx:
            return "", []

        chunk_ids = list(range(start_idx, end_idx))
        texts = []
        for cid in chunk_ids:
            src = self.sources[cid] if cid < len(self.sources) else "unknown"
            texts.append(f"--- Chunk {cid} (source: {src}) ---\n{self.chunks[cid]}")

        return "\n\n".join(texts), chunk_ids

    def format_result(self, text: str, chunk_ids: List[int]) -> str:
        """Format for agent consumption."""
        if not text:
            return "ERROR: No content found for specified range."
        return f"Content from chunks {chunk_ids}:\n\n{text}"


class GrepSearch:
    """Regex search across raw markdown content."""

    def __init__(self, markdown_path: str):
        self.markdown_path = Path(markdown_path)
        self._lines: Optional[List[str]] = None

    def _load_lines(self) -> List[str]:
        if self._lines is None:
            with open(self.markdown_path, "r", encoding="utf-8") as f:
                self._lines = f.readlines()
        return self._lines

    def grep_text(
        self,
        pattern: str,
        context_lines: int = 2,
        max_matches: int = 10,
    ) -> List[GrepMatch]:
        """
        Search for pattern in markdown file.
        Returns matches with surrounding context.
        """
        lines = self._load_lines()
        try:
            compiled = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern: {pattern} ({e})")

        matches = []

        for i, line in enumerate(lines):
            if compiled.search(line):
                start = max(0, i - context_lines)
                end = min(len(lines), i + context_lines + 1)
                matches.append(
                    GrepMatch(
                        line_number=i + 1,
                        content=line.rstrip(),
                        context_before=[l.rstrip() for l in lines[start:i]],
                        context_after=[l.rstrip() for l in lines[i + 1 : end]],
                    )
                )
                if len(matches) >= max_matches:
                    break

        return matches

    def format_result(self, matches: List[GrepMatch]) -> str:
        """Format grep results for agent."""
        if not matches:
            return f"No matches found for pattern."
        lines = [f"Found {len(matches)} matches:"]
        for m in matches:
            lines.append(f"\n  Line {m.line_number}: {m.content}")
            if m.context_before:
                for ctx in m.context_before[-2:]:
                    lines.append(f"    (before) {ctx}")
            if m.context_after:
                for ctx in m.context_after[:2]:
                    lines.append(f"    (after) {ctx}")
        return "\n".join(lines)


class SectionSummarizer:
    """Retrieve section summaries from generated summaries file."""

    def __init__(self, summaries_path: str):
        self.summaries_path = Path(summaries_path)
        self._summaries: Optional[List[Dict]] = None

    def _load_summaries(self) -> List[Dict]:
        if self._summaries is None:
            if not self.summaries_path.exists():
                raise FileNotFoundError(
                    f"Summaries file not found: {self.summaries_path}\n"
                    "Run: python -m src.agent.generate_summaries"
                )
            with open(self.summaries_path, "r", encoding="utf-8") as f:
                self._summaries = json.load(f)
        return self._summaries

    def get_section_summary(self, section_name: str) -> Optional[Dict]:
        """Find section by name and return its summary."""
        summaries = self._load_summaries()
        section_name_lower = section_name.lower()

        for summ in summaries:
            heading = summ.get("heading", "")
            if section_name_lower in heading.lower():
                return {
                    "heading": heading,
                    "summary": summ.get("summary", ""),
                    "content_length": summ.get("content_length", 0),
                }
        return None

    def list_sections(self, limit: int = 30) -> List[str]:
        """List available section headings with summaries."""
        summaries = self._load_summaries()
        results = []
        for s in summaries[:limit]:
            heading = s.get("heading", "Untitled")
            summary = s.get("summary", "")
            if summary:
                results.append(f"{heading}: {summary[:100]}")
            else:
                results.append(heading)
        return results

    def format_result(self, result: Optional[Dict]) -> str:
        """Format section summary for agent."""
        if result is None:
            return "ERROR: Section not found. Use list_sections to see available sections."
        return f"Section: {result['heading']}\nSummary: {result['summary']}\nFull length: {result['content_length']} chars"


class AgentToolkit:
    """Container for all agent tools, initialized from artifacts."""

    def __init__(
        self,
        faiss_index: faiss.Index,
        chunks: List[str],
        sources: List[str],
        embed_model: str,
        markdown_path: str,
        summaries_path: str,
    ):
        self.index_scout = IndexScout(faiss_index, chunks, sources, embed_model)
        self.reader = NavigationalReader(chunks, sources)
        self.grep = GrepSearch(markdown_path)
        self.summarizer = SectionSummarizer(summaries_path)

    def execute(self, tool_name: str, tool_args: Dict) -> str:
        """Execute a tool by name with given arguments."""
        try:
            if tool_name == "search_index":
                results = self.index_scout.search_index(
                    query=tool_args["query"],
                    top_k=tool_args.get("top_k", 10),
                )
                return self.index_scout.format_result(results)

            elif tool_name == "read_content":
                text, chunk_ids = self.reader.read_content(
                    target_chunk_id=tool_args["target_chunk_id"],
                    relative_start=tool_args.get("relative_start", 0),
                    relative_end=tool_args.get("relative_end", 0),
                )
                return self.reader.format_result(text, chunk_ids)

            elif tool_name == "grep_text":
                matches = self.grep.grep_text(
                    pattern=tool_args["pattern"],
                    context_lines=tool_args.get("context_lines", 2),
                    max_matches=tool_args.get("max_matches", 10),
                )
                return self.grep.format_result(matches)

            elif tool_name == "get_section_summary":
                result = self.summarizer.get_section_summary(
                    section_name=tool_args["section_name"]
                )
                return self.summarizer.format_result(result)

            elif tool_name == "list_sections":
                sections = self.summarizer.list_sections(
                    limit=tool_args.get("limit", 30)
                )
                return "Available sections:\n" + "\n".join(sections)

            else:
                return f"ERROR: Unknown tool '{tool_name}'. Available: search_index, read_content, grep_text, get_section_summary, list_sections"

        except Exception as e:
            return f"ERROR executing {tool_name}: {type(e).__name__}: {str(e)}"

    @staticmethod
    def get_tool_descriptions() -> str:
        """Return tool descriptions for the agent prompt."""
        return """Available tools:

1. search_index(query: str, top_k: int = 10)
   - Returns: Structured list with chunk_id, score, source, preview
   - Use chunk_id from results for read_content
   - Best for: Finding relevant content by semantic similarity

2. read_content(target_chunk_id: int, relative_start: int = 0, relative_end: int = 0)
   - Returns: Full text of specified chunk range
   - relative_start=-1, relative_end=1 reads 3 chunks (before, target, after)
   - Use chunk_id from search_index results

3. grep_text(pattern: str, context_lines: int = 2, max_matches: int = 10)
   - Returns: Line numbers and matches with context
   - Use for: Exact terms, code snippets, specific phrases
   - Pattern is case-insensitive regex

4. get_section_summary(section_name: str)
   - Returns: AI-generated summary of section
   - Use for: Quick overview before reading full content
   - Partial match on section heading

5. list_sections(limit: int = 30)
   - Returns: Available section headings with brief summaries
   - Use for: Understanding document structure"""

