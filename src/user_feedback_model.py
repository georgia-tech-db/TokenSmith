import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set

from src.retriever import IndexKeywordRetriever


class TopicExtractor:
    def __init__(
        self,
        extracted_index_path,
        page_to_chunk_map_path,
        extracted_sections_path: Optional[Path] = None,
    ):
        self._ikr = IndexKeywordRetriever(
            extracted_index_path=extracted_index_path,
            page_to_chunk_map_path=page_to_chunk_map_path,
        )
        self._heading_tokens: Dict[str, Set[str]] = {}
        self._load_headings(extracted_sections_path or Path("data/extracted_sections.json"))

    def extract_topics(self, query: str, max_topics: int = 3) -> List[str]:
        keywords = self._ikr._extract_keywords(query)
        topics = self._extract_index_topics(keywords, max_topics)
        if topics:
            return topics
        return self._extract_heading_topics(query, max_topics)

    def _extract_index_topics(self, keywords: List[str], max_topics: int) -> List[str]:
        if not keywords:
            return []
        phrase_counts = Counter()
        for kw in keywords:
            for phrase in self._ikr.token_to_phrases.get(kw, []):
                phrase_counts[phrase] += 1
        topics = [p for p, _ in phrase_counts.most_common(max_topics)]
        return topics if topics else keywords[:max_topics]

    def _load_headings(self, extracted_sections_path: Path) -> None:
        if not extracted_sections_path.exists():
            return
        try:
            with open(extracted_sections_path, "r", encoding="utf-8") as f:
                sections = json.load(f)
        except (json.JSONDecodeError, OSError):
            return

        for item in sections:
            heading = item.get("heading")
            if not heading:
                continue
            tokens = self._heading_tokens_for(heading)
            if tokens:
                self._heading_tokens[heading] = tokens

    def _extract_heading_topics(self, query: str, max_topics: int) -> List[str]:
        query_tokens = self._heading_tokens_for(query)
        if not query_tokens:
            return []
        scores = []
        for heading, tokens in self._heading_tokens.items():
            overlap = len(tokens & query_tokens)
            if overlap:
                scores.append((heading, overlap))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [heading for heading, _ in scores[:max_topics]]

    @staticmethod
    def _heading_tokens_for(text: str) -> Set[str]:
        tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
        return {t for t in tokens if len(t) > 2}


def estimate_difficulty(text: str) -> str:
    tokens = IndexKeywordRetriever._extract_keywords(text)
    if not tokens:
        return "medium"

    if len(tokens) > 18:
        return "hard"
    if len(tokens) > 10:
        return "medium"
    return "easy"
