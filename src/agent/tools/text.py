import re
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

@dataclass
class GrepMatch:
    line_number: int
    content: str
    context_before: List[str]
    context_after: List[str]

class GrepSearch:
    """Regex search across raw markdown."""

    def __init__(self, markdown_path: Path):
        self.markdown_path = markdown_path
        self._lines: Optional[List[str]] = None

    def _load_lines(self) -> List[str]:
        if self._lines is None:
            with open(self.markdown_path, "r", encoding="utf-8") as f:
                self._lines = f.readlines()
        return self._lines

    def search(self, pattern: str, context_lines: int = 2, max_matches: int = 10) -> List[GrepMatch]:
        lines = self._load_lines()
        try:
            compiled = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            raise ValueError(f"Invalid regex: {pattern} ({e})")

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
        if not matches:
            return "No matches found."
        lines = [f"Found {len(matches)} matches:"]
        for m in matches:
            lines.append(f"\n  Line {m.line_number}: {m.content}")
            for ctx in m.context_before[-2:]:
                lines.append(f"    (before) {ctx}")
            for ctx in m.context_after[:2]:
                lines.append(f"    (after) {ctx}")
        return "\n".join(lines)
