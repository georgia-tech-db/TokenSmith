from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence


CLEANING_PROFILE_VERSION = 2
DEFAULT_CLEANING_PROFILE_ID = "course"

CLEANING_PROFILES: Dict[str, Dict[str, str]] = {
    "course": {
        "id": "course",
        "name": "Textbook",
        "description": "Repairs wrapped lines, removes repeated page headers, and records chapter or section headings.",
    },
    "article": {
        "id": "article",
        "name": "Article",
        "description": "Builds cleaner paragraphs for web articles, encyclopedia PDFs, and reports.",
    },
    "minimal": {
        "id": "minimal",
        "name": "Minimal",
        "description": "Only normalizes spacing and blank lines.",
    },
}

CLEANING_RULES: Dict[str, Dict[str, Any]] = {
    "normalize_text": {
        "id": "normalize_text",
        "name": "Normalize text",
        "description": "Removes null characters and normalizes spaces, line endings, and blank lines.",
        "locked": True,
    },
    "remove_repeated_edges": {
        "id": "remove_repeated_edges",
        "name": "Remove repeated page headers and footers",
        "description": "Drops repeated edge lines that appear across multiple pages.",
    },
    "repair_hyphenated_breaks": {
        "id": "repair_hyphenated_breaks",
        "name": "Repair hyphenated line breaks",
        "description": "Joins words split across PDF line breaks.",
    },
    "merge_wrapped_lines": {
        "id": "merge_wrapped_lines",
        "name": "Merge wrapped paragraph lines",
        "description": "Builds paragraphs from lines that were wrapped by PDF extraction.",
    },
    "detect_article_section_headers": {
        "id": "detect_article_section_headers",
        "name": "Detect article section headers",
        "description": "Records common wiki/article headings such as Career, References, and External links as chunk context.",
    },
    "detect_chapter_section_headers": {
        "id": "detect_chapter_section_headers",
        "name": "Detect chapter section headers",
        "description": "Records textbook headings such as Chapter 12 or 12.1 Storage as chunk context.",
    },
}

PROFILE_RULE_IDS: Dict[str, List[str]] = {
    "course": [
        "normalize_text",
        "remove_repeated_edges",
        "repair_hyphenated_breaks",
        "merge_wrapped_lines",
        "detect_chapter_section_headers",
    ],
    "article": [
        "normalize_text",
        "remove_repeated_edges",
        "repair_hyphenated_breaks",
        "merge_wrapped_lines",
        "detect_article_section_headers",
    ],
    "minimal": ["normalize_text"],
}

ARTICLE_SECTION_HEADER_KEYS = {
    "abstract",
    "background",
    "biography",
    "career",
    "early life",
    "early life and education",
    "education",
    "external links",
    "history",
    "overview",
    "personal life",
    "political career",
    "political positions",
    "presidency of the house of representatives and disy",
    "public image",
    "references",
}


def cleaning_profiles() -> List[Dict[str, Any]]:
    return [
        {**profile, "defaultRuleIds": PROFILE_RULE_IDS.get(profile["id"], PROFILE_RULE_IDS[DEFAULT_CLEANING_PROFILE_ID])}
        for profile in CLEANING_PROFILES.values()
    ]


def cleaning_rules() -> List[Dict[str, Any]]:
    return list(CLEANING_RULES.values())


def resolve_cleaning_profile(profile_id: Any) -> Dict[str, str]:
    key = str(profile_id or DEFAULT_CLEANING_PROFILE_ID).strip()
    return CLEANING_PROFILES.get(key) or CLEANING_PROFILES[DEFAULT_CLEANING_PROFILE_ID]


def default_rule_ids_for_profile(profile_id: Any) -> List[str]:
    profile = resolve_cleaning_profile(profile_id)
    return list(PROFILE_RULE_IDS.get(profile["id"], PROFILE_RULE_IDS[DEFAULT_CLEANING_PROFILE_ID]))


def resolve_cleaning_rule_ids(profile_id: Any, rule_ids: Optional[Sequence[Any]] = None) -> List[str]:
    if rule_ids is None:
        raw_ids = default_rule_ids_for_profile(profile_id)
    elif isinstance(rule_ids, str):
        raw_ids = [rule_ids]
    else:
        raw_ids = [str(rule_id) for rule_id in rule_ids]
    known_ids = set(CLEANING_RULES.keys())
    selected = [rule_id for rule_id in raw_ids if rule_id in known_ids]
    if "normalize_text" not in selected:
        selected.insert(0, "normalize_text")

    ordered = []
    for rule_id in CLEANING_RULES:
        if rule_id in selected:
            ordered.append(rule_id)
    return ordered


def normalize_extracted_text(text: str) -> str:
    value = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"[ \t]+", " ", value)
    value = "\n".join(line.rstrip() for line in value.splitlines())
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def normalized_line_key(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip().lower()


def repeated_edge_lines(pages: List[Dict[str, Any]]) -> set[str]:
    if len(pages) < 2:
        return set()

    counts: Dict[str, int] = {}
    for page in pages:
        lines = [line.strip() for line in str(page.get("text") or "").splitlines() if line.strip()]
        edge_lines = [*lines[:3], *lines[-3:]]
        for line in set(edge_lines):
            key = normalized_line_key(line)
            if 4 <= len(key) <= 140:
                counts[key] = counts.get(key, 0) + 1

    threshold = max(2, round(len(pages) * 0.6))
    return {key for key, count in counts.items() if count >= threshold}


def remove_repeated_edges(text: str, repeated_lines: set[str]) -> str:
    if not repeated_lines:
        return text

    kept = []
    for line in text.splitlines():
        key = normalized_line_key(line)
        if key and key in repeated_lines:
            continue
        kept.append(line)
    return "\n".join(kept)


def repair_hyphenated_breaks(text: str) -> str:
    return re.sub(r"([A-Za-z])-\n([A-Za-z])", r"\1\2", text)


def is_list_line(line: str) -> bool:
    return bool(re.match(r"^\s*(?:[-*•▪]|\d+[.)])\s+", line))


def is_probable_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if len(stripped) > 90:
        return False
    if is_list_line(stripped):
        return False
    if stripped.endswith((".", ",", ";", ":")):
        return False
    words = stripped.split()
    if len(words) > 10:
        return False
    if stripped.isupper() or re.match(r"^(?:chapter|section)\s+\d+", stripped, re.IGNORECASE):
        return True

    lower_heading_words = {"a", "an", "and", "as", "for", "in", "of", "on", "or", "the", "to", "with"}
    heading_like_words = 0
    for word in words:
        normalized = re.sub(r"[^A-Za-z0-9]", "", word)
        if not normalized:
            continue
        if normalized.lower() in lower_heading_words or normalized[:1].isupper() or normalized.isupper():
            heading_like_words += 1

    return bool(words) and heading_like_words / len(words) >= 0.75


def normalize_section_header(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip(" =#\t")


def section_header_from_line(line: str, rule_ids: Optional[Sequence[Any]] = None) -> Optional[str]:
    selected_rules = {str(rule_id) for rule_id in (rule_ids or [])}
    stripped = normalize_section_header(line)

    if not stripped or len(stripped) > 120 or is_list_line(stripped):
        return None

    if "detect_article_section_headers" in selected_rules:
        markdown_heading = re.match(r"^#{1,6}\s+(.+?)\s*$", line.strip())
        mediawiki_heading = re.match(r"^={2,6}\s*(.+?)\s*={2,6}$", line.strip())
        if markdown_heading:
            return normalize_section_header(markdown_heading.group(1))
        if mediawiki_heading:
            return normalize_section_header(mediawiki_heading.group(1))
        if normalized_line_key(stripped) in ARTICLE_SECTION_HEADER_KEYS:
            return stripped

    if "detect_chapter_section_headers" in selected_rules:
        if re.match(r"^(?:chapter|section)\s+\d+\b(?:\s*[:.-]\s*\S.*)?$", stripped, re.IGNORECASE):
            return stripped
        if re.match(r"^\d+(?:\.\d+)+\s+\S.{0,100}$", stripped):
            return stripped

    return None


def paragraphize_text(text: str, *, preserve_short_lines: bool = False) -> str:
    paragraphs: List[str] = []
    current: List[str] = []

    def flush() -> None:
        nonlocal current
        if current:
            paragraphs.append(" ".join(current).strip())
            current = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            flush()
            continue
        if preserve_short_lines and (is_list_line(line) or len(line) <= 70):
            flush()
            paragraphs.append(line)
            continue
        if is_list_line(line) or is_probable_heading(line):
            flush()
            paragraphs.append(line)
            continue
        current.append(line)

    flush()
    return "\n\n".join(paragraph for paragraph in paragraphs if paragraph)


def clean_page_text(
    text: str,
    profile_id: Any = None,
    *,
    repeated_lines: set[str] | None = None,
    rule_ids: Optional[Sequence[Any]] = None,
) -> str:
    profile = resolve_cleaning_profile(profile_id)
    selected_rules = set(resolve_cleaning_rule_ids(profile["id"], rule_ids))
    cleaned = normalize_extracted_text(text) if "normalize_text" in selected_rules else text

    if "remove_repeated_edges" in selected_rules and repeated_lines:
        cleaned = remove_repeated_edges(cleaned, repeated_lines)

    if "repair_hyphenated_breaks" in selected_rules:
        cleaned = repair_hyphenated_breaks(cleaned)

    if "merge_wrapped_lines" in selected_rules:
        cleaned = paragraphize_text(cleaned)

    return normalize_extracted_text(cleaned) if "normalize_text" in selected_rules else cleaned.strip()


def clean_pages(
    pages: List[Dict[str, Any]],
    profile_id: Any = None,
    rule_ids: Optional[Sequence[Any]] = None,
) -> List[Dict[str, Any]]:
    profile = resolve_cleaning_profile(profile_id)
    selected_rules = resolve_cleaning_rule_ids(profile["id"], rule_ids)
    repeated_lines = repeated_edge_lines(pages) if "remove_repeated_edges" in selected_rules else set()
    cleaned_pages: List[Dict[str, Any]] = []

    for page in pages:
        cleaned = clean_page_text(
            page.get("text") or "",
            profile["id"],
            repeated_lines=repeated_lines,
            rule_ids=selected_rules,
        )
        if cleaned:
            cleaned_pages.append({**page, "text": cleaned})

    return cleaned_pages
