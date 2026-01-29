import json
from pathlib import Path
from typing import List, Dict, Optional

class SectionSummarizer:
    """Retrieve section summaries from pre-generated file."""

    def __init__(self, summaries_path: Path):
        self.summaries_path = summaries_path
        self._summaries: Optional[List[Dict]] = None

    def _load_summaries(self) -> List[Dict]:
        if self._summaries is None:
            with open(self.summaries_path, "r", encoding="utf-8") as f:
                self._summaries = json.load(f)
        return self._summaries

    def get_summary(self, section_name: str) -> Optional[Dict]:
        summaries = self._load_summaries()
        section_lower = section_name.lower()
        for s in summaries:
            heading = s.get("heading", "")
            if section_lower in heading.lower():
                return {
                    "heading": heading,
                    "summary": s.get("summary", ""),
                    "content_length": s.get("content_length", 0),
                }
        return None

    def list_sections(self, limit: int = 30) -> List[str]:
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
        if result is None:
            return "Section not found."
        return f"Section: {result['heading']}\nSummary: {result['summary']}\nLength: {result['content_length']} chars"
