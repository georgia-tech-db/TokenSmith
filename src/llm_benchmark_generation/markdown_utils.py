"""
src/llm_benchmark_generation/markdown_utils.py

Utilities for loading and slicing the textbook markdown by page number.
Page markers in the markdown have the form:  --- Page N ---
The marker appears at the TOP of page N, so page N's content runs from
that marker up to (but not including) the --- Page N+1 --- marker.
"""

from __future__ import annotations

import re
import pathlib
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Markdown loading
# ─────────────────────────────────────────────────────────────────────────────

def load_markdown(path: str | pathlib.Path) -> tuple[str, dict[int, int]]:
    """
    Load the full markdown file and build a page-number → character-offset index.

    Also performs two cleaning passes before indexing:
      - Removes <!-- image --> placeholders (add noise with no informational value)
      - Collapses runs of 3+ newlines to 2 (one blank line between paragraphs)

    Returns
    -------
    (full_text, page_offsets)
    page_offsets maps page_number (int) → character start offset in full_text.
    """
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Markdown not found: {path.resolve()}")

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # Clean image placeholders
    text = re.sub(r"<!--\s*image\s*-->", "", text)
    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Build page offset index
    offsets: dict[int, int] = {}
    for m in re.finditer(r"--- Page (\d+) ---", text):
        offsets[int(m.group(1))] = m.start()

    if not offsets:
        raise ValueError(
            f"No '--- Page N ---' markers found in {path}. "
            "Is this the right markdown file?"
        )

    return text, offsets


# ─────────────────────────────────────────────────────────────────────────────
# Page extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_pages(
    full_text: str,
    offsets:   dict[int, int],
    start:     int,
    end:       int,
) -> str:
    """
    Return the raw markdown text for pages [start, end] inclusive.

    Raises ValueError if start page is not in the offset index.
    """
    if start not in offsets:
        raise ValueError(
            f"Page {start} not found in markdown. "
            f"Available range: {min(offsets)}–{max(offsets)}"
        )

    begin_char = offsets[start]
    max_page   = max(offsets.keys())

    # Find the first page AFTER our range that exists in the index
    next_page = end + 1
    while next_page <= max_page and next_page not in offsets:
        next_page += 1

    end_char = offsets.get(next_page, len(full_text))
    return full_text[begin_char:end_char]


# ─────────────────────────────────────────────────────────────────────────────
# Windowing
# ─────────────────────────────────────────────────────────────────────────────

def get_page_windows(
    chapter_start: int,
    chapter_end:   int,
    window_size:   int = 25,
) -> list[tuple[int, int]]:
    """
    Split a chapter page range into windows of exactly window_size pages.

    Strategy:
      - Walk from chapter_start in non-overlapping steps of window_size.
      - If the final remaining segment is shorter than window_size, extend
        it BACKWARDS so the last window is always exactly window_size pages.
        This means the last window may overlap with the second-to-last.
      - Every window always ends exactly at chapter_end.

    Returns a list of (window_start, window_end) tuples.
    """
    if chapter_end - chapter_start + 1 <= window_size:
        # Entire chapter fits in one window
        return [(chapter_start, chapter_end)]

    windows: list[tuple[int, int]] = []
    pos = chapter_start

    while pos <= chapter_end:
        win_end = min(pos + window_size - 1, chapter_end)

        # If tail is shorter than window_size, back-fill
        if (win_end - pos + 1) < window_size and win_end == chapter_end:
            pos     = max(chapter_start, chapter_end - window_size + 1)
            win_end = chapter_end

        windows.append((pos, win_end))

        if win_end == chapter_end:
            break
        pos += window_size

    return windows


# ─────────────────────────────────────────────────────────────────────────────
# Book info loading
# ─────────────────────────────────────────────────────────────────────────────

def load_book_info(book_info_path: str | pathlib.Path) -> Optional[dict]:
    """
    Load the book_chapters_page_info.json file.
    Returns the parsed dict or None if the file is missing or malformed.
    """
    path = pathlib.Path(book_info_path)
    if not path.exists():
        print(f"  [WARN] Book info file not found: {path}")
        return None
    try:
        import json
        with open(path) as f:
            data = json.load(f)
        return data
    except Exception as exc:
        print(f"  [WARN] Could not load book info: {exc}")
        return None


def resolve_chapters(
    book_info:     Optional[dict],
    default_md:    str,
) -> tuple[str, dict[int, dict]]:
    """
    Resolve the markdown path and chapter boundaries from book_info.

    If book_info is None or contains no chapter data, returns the full
    markdown treated as a single chapter spanning all found pages.

    Returns
    -------
    (markdown_path, chapters_dict)
    chapters_dict maps chapter_number (int) →
        {"content_start": int, "content_end": int}
    """
    md_path = default_md

    if book_info:
        md_path = book_info.get("markdown_path", default_md)
        raw_chapters = book_info.get("chapters", {})
        if raw_chapters:
            chapters = {
                int(k): {
                    "content_start": v["content_start"],
                    "content_end":   v["content_end"],
                }
                for k, v in raw_chapters.items()
            }
            return md_path, chapters

    # Fallback: load markdown and treat as one chapter
    print("  [INFO] No chapter boundaries found — treating full markdown as one chapter")
    _, offsets = load_markdown(md_path)
    if not offsets:
        raise ValueError("No page markers found in the markdown file.")
    first = min(offsets.keys())
    last  = max(offsets.keys())
    return md_path, {1: {"content_start": first, "content_end": last}}