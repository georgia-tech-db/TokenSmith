"""
src/llm_benchmark_generation/verification.py

Substring verification for gold chunks against the source markdown pages.

Four-tier matching strategy (stops at first hit per chunk):
  1. Exact normalised match on raw source
  2. Exact normalised match on cleaned source  (handles mid-sentence page markers)
  3. Trailing-punct-stripped match on raw source  (handles sentence-end mismatches)
  4. Trailing-punct-stripped match on cleaned source  (combined edge case)

The cleaning step removes:
  - Lines that consist entirely of symbols  (stray ';', '---', etc.)
  - '--- Page N ---' marker lines
  - Standalone 'Page N' artifacts (printed page numbers)
"""

from __future__ import annotations

import re


# ─────────────────────────────────────────────────────────────────────────────
# Text normalisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def normalise_ws(text: str) -> str:
    """Collapse all whitespace runs to a single space and strip edges."""
    return re.sub(r"\s+", " ", text).strip()


def clean_md_for_verification(text: str) -> str:
    """
    Remove markdown artifacts that can interrupt mid-sentence substring matches.

    Removes in order:
      1. Lines containing only punctuation / symbols  (e.g. a stray ';' line)
      2. Page marker lines:  '--- Page N ---'
      3. Standalone page-number artifacts:  'Page N'

    After removal, whitespace is normalised so sentences split across page
    boundaries become one continuous string.
    """
    # Remove lines that are purely symbolic (no alphanumeric characters)
    text = re.sub(r"^\s*[\.,!?;:\-_]+\s*$", " ", text, flags=re.MULTILINE)
    # Remove --- Page N --- markers
    text = re.sub(r"---\s*Page\s+\d+\s*---", " ", text)
    # Remove standalone 'Page N' artifacts
    text = re.sub(r"\bPage\s+\d+\b", " ", text)
    return normalise_ws(text)


# ─────────────────────────────────────────────────────────────────────────────
# Per-chunk check
# ─────────────────────────────────────────────────────────────────────────────

def check_chunk_in_pages(chunk: str, pages_text: str) -> bool:
    """
    Return True if chunk is found verbatim (after normalisation) in pages_text.
    Applies the four-tier matching strategy described in the module docstring.
    """
    norm_source   = normalise_ws(pages_text)
    clean_source  = clean_md_for_verification(pages_text)
    norm_chunk    = normalise_ws(chunk)
    stripped_chunk = norm_chunk.rstrip(".,;:!?")

    return (
        norm_chunk     in norm_source
        or norm_chunk  in clean_source
        or stripped_chunk in norm_source
        or stripped_chunk in clean_source
    )


# ─────────────────────────────────────────────────────────────────────────────
# Full record check
# ─────────────────────────────────────────────────────────────────────────────

def verify_gold_chunks(record: dict, pages_text: str) -> dict:
    """
    Check every gold chunk in record against pages_text.

    Returns
    -------
    {
        "passed":   bool,
        "failures": [list of chunk strings that did not match]
    }
    """
    failures = [
        chunk
        for chunk in record.get("gold_chunks", [])
        if not check_chunk_in_pages(chunk, pages_text)
    ]
    return {
        "passed":   len(failures) == 0,
        "failures": failures,
    }