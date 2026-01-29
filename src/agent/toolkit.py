from typing import List, Dict, Tuple, Optional
from pathlib import Path
import faiss

from src.agent.tools.search import IndexScout
from src.agent.tools.read import NavigationalReader
from src.agent.tools.text import GrepSearch
from src.agent.tools.sections import SectionSummarizer

class AgentToolkit:
    """Container for agent tools."""

    def __init__(
        self,
        faiss_index: faiss.Index,
        chunks: List[str],
        sources: List[str],
        embed_model: str,
        markdown_path: Optional[str] = None,
        summaries_path: Optional[str] = None,
    ):
        self.index_scout = IndexScout(faiss_index, chunks, sources, embed_model)
        self.reader = NavigationalReader(chunks, sources)
        self.grep: Optional[GrepSearch] = None
        self.summarizer: Optional[SectionSummarizer] = None

        if markdown_path and Path(markdown_path).exists():
            self.grep = GrepSearch(Path(markdown_path))

        if summaries_path and Path(summaries_path).exists():
            self.summarizer = SectionSummarizer(Path(summaries_path))

        self._available_tools = ["search_index", "read_content"]
        if self.grep: self._available_tools.append("grep_text")
        if self.summarizer: self._available_tools.extend(["get_section_summary", "list_sections"])

    @property
    def available_tools(self) -> List[str]:
        return self._available_tools

    def get_tool_descriptions(self) -> str:
        # Construct dynamic tool descriptions based on availability
        base = """
Available Tools:
- `search_index(query="...")`: Semantic search. Returns relevant chunk IDs vs preview text.
- `read_content(target_chunk_id=123, relative_start=-1, relative_end=1)`: Read full text of chunks. Use relative_start/end to read surrounding context.
"""
        if self.grep:
            base += '- `grep_text(pattern="regex")`: Search for exact patterns in the full text.\n'
        if self.summarizer:
            base += '- `list_sections(limit=30)`: List available document sections.\n'
            base += '- `get_section_summary(section_name="...")`: Get summary of a specific section.\n'
        return base

    def get_initial_context(self, question: str, top_k: int = 5) -> Tuple[str, bool]:
        """Helper to get initial search results."""
        results = self.index_scout.search(question, top_k=top_k)
        return self.index_scout.format_result(results), True

    def execute(self, tool_name: str, tool_args: Dict) -> Tuple[str, bool]:
        if tool_name not in self._available_tools:
            return f"Unknown tool '{tool_name}'. Available: {', '.join(self._available_tools)}", False

        try:
            if tool_name == "search_index":
                return self.index_scout.format_result(
                    self.index_scout.search(tool_args.get("query"), top_k=tool_args.get("top_k", 10))
                ), True
            
            elif tool_name == "read_content":
                text, chunk_ids = self.reader.read(
                    tool_args.get("target_chunk_id"),
                    tool_args.get("relative_start", 0),
                    tool_args.get("relative_end", 0),
                )
                return self.reader.format_result(text, chunk_ids), True
            
            elif tool_name == "grep_text" and self.grep:
                return self.grep.format_result(
                    self.grep.search(tool_args.get("pattern"), tool_args.get("context_lines", 2))
                ), True
            
            elif tool_name == "list_sections" and self.summarizer:
                return "\n".join(self.summarizer.list_sections(tool_args.get("limit", 30))), True
            
            elif tool_name == "get_section_summary" and self.summarizer:
                return self.summarizer.format_result(
                    self.summarizer.get_summary(tool_args.get("section_name", ""))
                ), True

        except Exception as e:
            return f"Tool execution failed: {str(e)}", False
            
        return "Tool not available or arguments invalid.", False
