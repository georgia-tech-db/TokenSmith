"""Textbook vocabulary and query typo correction.

Off by default; controlled by the "Typo Correction" toggle in app settings (see
``TYPO_CORRECTION_ENABLED`` in tokensmith_engine.py). When enabled, indexing
records a per-material list of high-frequency words from the cleaned document
text, and retrieval rewrites query tokens that look like typos to their closest
vocabulary match (``difflib`` with a similarity cutoff of 0.86 by default).

The vocabulary for a material is stored as a single JSON file at
``<userData>/vocab/<materialId>.json`` so it is easy to inspect or delete during
development and needs no database migration.
"""

from __future__ import annotations

import difflib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

VOCAB_DIR_NAME = "vocab"
VOCAB_FORMAT_VERSION = 1

# difflib produces too many false matches on short strings, so only tokens at
# least this long are considered for correction.
MIN_TOKEN_LENGTH_FOR_CORRECTION = 4
# Words shorter than this are never added to the vocabulary.
MIN_VOCAB_WORD_LENGTH = 3
# Keep the vocabulary bounded so per-query difflib scans stay fast.
MAX_VOCAB_WORDS = 5000
DEFAULT_MIN_COUNT = 4
DEFAULT_TYPO_CUTOFF = 0.86

# Mirrors STOP_WORDS in tokensmith_engine; duplicated here to keep this module
# free of engine imports (the engine imports this module, not the other way).
_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "does", "for", "from", "how",
    "i", "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "was",
    "what", "when", "where", "which", "who", "why", "with",
}

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*[A-Za-z]|[A-Za-z]")


def _vocab_tokens(text: str) -> List[str]:
    return [match.group(0).lower() for match in _WORD_RE.finditer(text or "")]


def build_vocabulary(
    chunk_texts: Iterable[str],
    *,
    min_count: int = DEFAULT_MIN_COUNT,
    max_words: int = MAX_VOCAB_WORDS,
) -> Dict[str, int]:
    """Count high-frequency alphabetic words across the given texts."""

    counts: "Counter[str]" = Counter()
    for text in chunk_texts:
        for token in _vocab_tokens(text):
            if len(token) < MIN_VOCAB_WORD_LENGTH or token in _STOP_WORDS:
                continue
            counts[token] += 1

    kept = [(word, count) for word, count in counts.items() if count >= max(1, min_count)]
    kept.sort(key=lambda item: (-item[1], item[0]))
    return dict(kept[:max_words])


def _safe_material_id(material_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(material_id)) or "material"


def vocab_path(user_data_path: str, material_id: str) -> Path:
    return Path(user_data_path) / VOCAB_DIR_NAME / f"{_safe_material_id(material_id)}.json"


def save_vocabulary(
    user_data_path: str,
    material_id: str,
    vocab: Dict[str, int],
    *,
    min_count: int = DEFAULT_MIN_COUNT,
) -> Path:
    path = vocab_path(user_data_path, material_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": VOCAB_FORMAT_VERSION,
        "materialId": str(material_id),
        "minCount": int(min_count),
        "wordCount": len(vocab),
        "words": vocab,
    }
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(path)
    return path


def load_vocabulary(user_data_path: str, material_id: str) -> Optional[Dict[str, int]]:
    path = vocab_path(user_data_path, material_id)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    words = data.get("words") if isinstance(data, dict) else None
    if not isinstance(words, dict):
        return None
    result: Dict[str, int] = {}
    for key, value in words.items():
        try:
            result[str(key)] = int(value)
        except (TypeError, ValueError):
            continue
    return result


def delete_vocabulary(user_data_path: str, material_id: str) -> None:
    try:
        vocab_path(user_data_path, material_id).unlink()
    except (FileNotFoundError, OSError):
        pass


def _match_case(source: str, replacement: str) -> str:
    if source[:1].isupper() and not source.isupper():
        return replacement[:1].upper() + replacement[1:]
    if source.isupper():
        return replacement.upper()
    return replacement


def correct_query(
    query: str,
    vocab_words: Sequence[str],
    *,
    cutoff: float = DEFAULT_TYPO_CUTOFF,
) -> Tuple[str, List[Dict[str, str]]]:
    """Rewrite query tokens that look like typos to their closest vocab match.

    Returns the (possibly unchanged) query and a de-duplicated list of
    ``{"from": ..., "to": ...}`` replacements that were applied.
    """

    if not query or not vocab_words:
        return query, []

    vocab_list = list(vocab_words)
    vocab_set = set(vocab_list)
    resolved: Dict[str, str] = {}
    applied: "Dict[Tuple[str, str], None]" = {}

    def _replace(match: "re.Match[str]") -> str:
        word = match.group(0)
        lower = word.lower()
        if len(lower) < MIN_TOKEN_LENGTH_FOR_CORRECTION or lower in vocab_set:
            return word
        if lower not in resolved:
            close = difflib.get_close_matches(lower, vocab_list, n=1, cutoff=cutoff)
            resolved[lower] = close[0] if close else lower
        corrected = resolved[lower]
        if corrected == lower:
            return word
        applied.setdefault((lower, corrected), None)
        return _match_case(word, corrected)

    corrected_query = _WORD_RE.sub(_replace, query)
    replacements = [{"from": pair[0], "to": pair[1]} for pair in applied]
    return corrected_query, replacements
