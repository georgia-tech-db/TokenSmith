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
    """Semantic search that returns metadata without full text."""

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
            preview = self.chunks[idx][:100].replace("\n", " ")
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
        """Format search results as readable text for the agent."""
        if not results:
            return "No results found."
        lines = []
        for r in results:
            lines.append(
                f"Chunk {r.chunk_id} (score={r.score:.3f}, source={r.source}): {r.preview}..."
            )
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
            texts.append(f"[Chunk {cid} | {src}]\n{self.chunks[cid]}")

        return "\n\n".join(texts), chunk_ids

    def format_result(self, text: str, chunk_ids: List[int]) -> str:
        """Format for agent consumption."""
        if not text:
            return "No content found for the specified range."
        return f"Read chunks {chunk_ids}:\n{text}"


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
        compiled = re.compile(pattern, re.IGNORECASE)
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
            return "No matches found."
        lines = []
        for m in matches:
            lines.append(f"Line {m.line_number}: {m.content}")
            if m.context_before:
                lines.append(f"  Before: {' | '.join(m.context_before[-2:])}")
            if m.context_after:
                lines.append(f"  After: {' | '.join(m.context_after[:2])}")
        return "\n".join(lines)


class SectionSummarizer:
    """Retrieve section content from extracted_sections.json."""

    def __init__(self, sections_path: str, max_chars: int = 1000):
        self.sections_path = Path(sections_path)
        self.max_chars = max_chars
        self._sections: Optional[List[Dict]] = None

    def _load_sections(self) -> List[Dict]:
        if self._sections is None:
            with open(self.sections_path, "r", encoding="utf-8") as f:
                self._sections = json.load(f)
        return self._sections

    def get_section_summary(self, section_name: str) -> Optional[Dict]:
        """
        Find section by name (case-insensitive partial match).
        Returns heading and truncated content.
        """
        sections = self._load_sections()
        section_name_lower = section_name.lower()

        for section in sections:
            heading = section.get("heading", "")
            if section_name_lower in heading.lower():
                content = section.get("content", "")
                return {
                    "heading": heading,
                    "content": content[: self.max_chars],
                    "full_length": len(content),
                }
        return None

    def list_sections(self, limit: int = 20) -> List[str]:
        """List available section headings."""
        sections = self._load_sections()
        return [s.get("heading", "Untitled") for s in sections[:limit]]

    def format_result(self, result: Optional[Dict]) -> str:
        """Format section result for agent."""
        if result is None:
            return "Section not found."
        truncated = "(truncated)" if result["full_length"] > self.max_chars else ""
        return f"{result['heading']}\n{result['content']} {truncated}"


class AgentToolkit:
    """Container for all agent tools, initialized from artifacts."""

    def __init__(
        self,
        faiss_index: faiss.Index,
        chunks: List[str],
        sources: List[str],
        embed_model: str,
        markdown_path: str,
        sections_path: str,
    ):
        self.index_scout = IndexScout(faiss_index, chunks, sources, embed_model)
        self.reader = NavigationalReader(chunks, sources)
        self.grep = GrepSearch(markdown_path)
        self.summarizer = SectionSummarizer(sections_path)

    def execute(self, tool_name: str, tool_args: Dict) -> str:
        """Execute a tool by name with given arguments."""
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
                limit=tool_args.get("limit", 20)
            )
            return "\n".join(sections)

        else:
            raise ValueError(f"Unknown tool: {tool_name}")

    @staticmethod
    def get_tool_descriptions() -> str:
        """Return tool descriptions for the agent prompt."""
        return """Available tools:

1. search_index(query: str, top_k: int = 10)
   - Semantic search returning chunk metadata (IDs, scores, sources, previews)
   - Use to find relevant sections before reading full content

2. read_content(target_chunk_id: int, relative_start: int = 0, relative_end: int = 0)
   - Read chunks with relative offsets from target
   - Example: target=100, start=-1, end=2 reads chunks 99-102

3. grep_text(pattern: str, context_lines: int = 2, max_matches: int = 10)
   - Regex search across raw markdown
   - Use for exact phrases, variable names, specific terms

4. get_section_summary(section_name: str)
   - Get section content by heading name (partial match)
   - Returns truncated content for overview

5. list_sections(limit: int = 20)
   - List available section headings"""

