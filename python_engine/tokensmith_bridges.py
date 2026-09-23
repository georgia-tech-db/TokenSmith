"""Bridge chunks: opt-in vocabulary-gap augmentation at index time.

Students ask in everyday words ("what happens when memory gets too full?"); textbooks answer in
their own terms ("buffer pool eviction"). With no shared words, retrieval never brings the right
passage into the window. This adds a small number of short passages written in student wording
that carry the book's actual mechanism, so a casual question has something to match.

Inventory (group chunks by source unit, else section header) -> audit (generated casual phrasings
that miss the top-K window) -> budget (a fraction of the chunk count) -> one bridge per miss.

Off by default. See docs/bridge-chunks.md for the measured tradeoff, including where it loses.
"""
from __future__ import annotations

import json
import math
import re
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

N_PHRASINGS = 2
K_WINDOW = 4
BUDGET_FRACTION = 0.05
MIN_BUDGET_CONCEPTS = 8         # floor so small materials still get useful coverage
MIN_CONCEPT_CHARS = 300
SKIP_KINDS = {"code", "table", "exercise", "question", "quiz"}  # nothing to bridge in these
_EXERCISE = re.compile(r"check your understanding|\bexercises?\b|\bquiz\b", re.IGNORECASE)
BRIDGE_KIND = "bridge"
BRIDGE_MIN_SCORE = 0.55         # a bridge must be this close (cosine) to the question to get a slot
BRIDGE_HEADER_PREFIX = "Bridge: "
# Commentary and preambles: a bridge is read as content, so any of these makes it worse than none.
_META = re.compile(
    r"\bhere(\'s| are| is)\b|\ban? (plain[- ]\w+ )?(explanation|summary|breakdown)\b|questions? a student"
    r"|\b(this|the) (passage|text|excerpt|author)\b|preamble",
    re.IGNORECASE,
)
_STOP = set("the a an is are was were this that of in on at to for with from by as and or not no if "
            "we you it its when how what why does do can be been".split())

EmbedFn = Callable[[Sequence[str]], np.ndarray]
GenFn = Callable[[str, float, int], str]


def tokens(text: str) -> List[str]:
    return [w for w in re.findall(r"[a-z0-9_]+", text.lower()) if len(w) > 2 and w not in _STOP]


def normalize_rows(vectors: Any) -> np.ndarray:
    """Row-normalize so a dot product is cosine similarity; callers may pass plain lists."""
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return matrix / norms


def build_inventory(
    chunks: Sequence[str],
    section_headers: Optional[Sequence[Optional[str]]] = None,
    chunk_kinds: Optional[Sequence[Optional[str]]] = None,
    unit_ids: Optional[Sequence[Optional[str]]] = None,
) -> Dict[str, int]:
    """Map concept -> index of its representative (longest) chunk.

    Source units are the retrieval atom, so they are the truest concept boundary when present.
    Prepared materials often carry empty or body-text section headers, which makes header
    grouping unreliable, so units win whenever the material has them.
    """
    carried: Optional[str] = None
    grouped: Dict[str, List[int]] = {}
    for index, text in enumerate(chunks):
        if chunk_kinds is not None and str(chunk_kinds[index] or "").lower() in SKIP_KINDS:
            continue
        headers = re.findall(r"^\s*#{1,6}\s+(.+?)\s*$", text, re.MULTILINE)
        explicit = (section_headers[index] if section_headers is not None else None) or (
            headers[-1].strip() if headers else None
        )
        if explicit:
            carried = explicit
        unit_id = unit_ids[index] if unit_ids is not None else None
        key = f"unit:{unit_id}" if unit_id else (explicit or carried)
        if key and not _EXERCISE.search(explicit or carried or ""):
            grouped.setdefault(key, []).append(index)
    return {
        key: representative
        for key, indexes in grouped.items()
        for representative in [max(indexes, key=lambda i: len(chunks[i]))]
        if len(chunks[representative]) >= MIN_CONCEPT_CHARS
    }


def generator_from_spec(model: Optional[Dict[str, Any]]) -> Optional[GenFn]:
    """A gen_fn for the app's selected generator, or None when it is not an Ollama model."""
    if not isinstance(model, dict) or model.get("engine") != "ollama":
        return None
    name = str(model.get("ollamaModelName") or "").strip()
    if not name:
        return None
    base = str(model.get("ollamaBaseUrl") or "").strip().rstrip("/") or "http://127.0.0.1:11434"

    def generate(prompt: str, temperature: float, num_predict: int) -> str:
        body = json.dumps({
            "model": name, "prompt": prompt[:6000], "stream": False,
            "options": {"temperature": temperature, "num_predict": num_predict},
        }).encode("utf-8")
        request = urllib.request.Request(
            f"{base}/api/generate", data=body,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(request, timeout=180) as response:
            return str(json.loads(response.read().decode("utf-8")).get("response") or "")

    return generate


def generate_phrasings(gen_fn: GenFn, chunk_text: str, count: int = N_PHRASINGS) -> List[str]:
    raw = gen_fn(
        "Here is a passage from a textbook.\n\n" + chunk_text[:900]
        + f"\n\nWrite {count} short questions a student might casually ask that this passage answers. "
        "Use everyday words, NOT the passage's technical terms. One per line, no numbering.",
        0.8, 100,
    )
    lines = [re.sub(r"^[\d.\-*\s]+", "", line).strip().rstrip("?") for line in raw.splitlines()]
    return [p for p in lines if 3 <= len(p.split()) <= 16 and not _META.search(p)][:count]


def generate_answer(gen_fn: GenFn, chunk_text: str, terms: Sequence[str]) -> str:
    """Plain-words explanation that must keep the passage's distinctive terms and must not talk
    about "the passage" (a bridge is read as content, not commentary). Empty when QC fails twice."""
    for _attempt in range(2):
        answer = gen_fn(
            "Here is a passage from a textbook.\n\n" + chunk_text[:900]
            + "\n\nExplain the mechanism described here in three or four plain sentences a beginner "
            f"would understand. You MUST keep these exact terms: {', '.join(terms[:5])}. "
            "Begin with the explanation itself: no preamble, and do not mention the passage, the text, "
            "or the author.",
            0.3, 170,
        ).strip()
        if sum(1 for term in terms if term in answer.lower()) >= 2 and not _META.search(answer):
            return answer
    return ""


def generate_bridges(
    chunks: Sequence[str],
    book_embeddings: np.ndarray,
    *,
    embed_fn: EmbedFn,
    gen_fn: GenFn,
    section_headers: Optional[Sequence[Optional[str]]] = None,
    chunk_kinds: Optional[Sequence[Optional[str]]] = None,
    unit_ids: Optional[Sequence[Optional[str]]] = None,
    budget_fraction: float = BUDGET_FRACTION,
    progress: Optional[Callable[[str], None]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Return (bridge chunk dicts ready for indexing, stats)."""
    inventory = build_inventory(chunks, section_headers, chunk_kinds, unit_ids)
    book_embeddings = normalize_rows(book_embeddings)
    total = len(chunks)
    document_frequency: Dict[str, int] = {}
    for text in chunks:
        for word in set(tokens(text)):
            document_frequency[word] = document_frequency.get(word, 0) + 1

    audit: Dict[str, Dict[str, Any]] = {}
    for position, (key, representative) in enumerate(inventory.items(), start=1):
        if progress:
            progress(f"Checking concept vocabulary ({position}/{len(inventory)})")
        phrasings = generate_phrasings(gen_fn, chunks[representative])
        misses, depths = [], []
        if phrasings:
            phrasing_matrix = normalize_rows(embed_fn(phrasings))
            for row, phrasing in enumerate(phrasings):
                order = np.argsort(-(book_embeddings @ phrasing_matrix[row]))
                rank = int(np.where(order == representative)[0][0]) + 1
                if rank > K_WINDOW:
                    misses.append(phrasing)
                    depths.append(rank)
        audit[key] = {
            "rep": representative, "misses": misses,
            "gap": len(misses) / len(phrasings) if phrasings else 0.0,
            "depth": float(np.mean(depths)) if depths else 0.0,
        }

    budget = max(MIN_BUDGET_CONCEPTS, int(budget_fraction * total))
    ranked = sorted(audit.items(), key=lambda item: (-item[1]["gap"], -item[1]["depth"]))
    selected = [(key, entry) for key, entry in ranked if entry["misses"]][:budget]

    bridges: List[Dict[str, Any]] = []
    for position, (key, entry) in enumerate(selected, start=1):
        if progress:
            progress(f"Writing bridge chunks ({position}/{len(selected)})")
        text = chunks[entry["rep"]]
        terms = sorted(set(tokens(text)), key=lambda w: -math.log(total / document_frequency.get(w, 1)))[:6]
        answer = generate_answer(gen_fn, text, terms)
        if not answer:
            continue
        # A unit key is a digest; prefer the chunk's own header for the source card people see.
        header = section_headers[entry["rep"]] if section_headers is not None else None
        label = (header or key.removeprefix("unit:")).strip()[:80]
        for phrasing in entry["misses"]:
            bridges.append({
                "text": f"Question: {phrasing}?\n{answer}",
                "sectionHeader": f"{BRIDGE_HEADER_PREFIX}{label}",
                "tokensmithChunkKind": BRIDGE_KIND,
                "sourceChunk": entry["rep"],
            })

    stats = {
        "concepts": len(inventory), "gapConcepts": sum(1 for e in audit.values() if e["misses"]),
        "budget": budget, "selected": len(selected), "bridges": len(bridges),
        "indexGrowth": round(len(bridges) / total, 3) if total else 0.0,
    }
    return bridges, stats


def splice_bridge_row(
    book_rows: Sequence[Dict[str, Any]],
    bridge_row: Optional[Dict[str, Any]],
    bridge_score: float,
    top_book_score: float,
    limit: int,
    min_score: float = BRIDGE_MIN_SCORE,
) -> List[Dict[str, Any]]:
    """Reserved-slot fusion. The book window is computed exactly as it would be without bridges, so
    bridges can never perturb it; then at most one bridge, chosen by vector similarity only, takes
    a slot: first if it beats the best book chunk, else last. Below `min_score` none is admitted,
    so shared everyday words cannot pull in a wrong-topic bridge."""
    rows = list(book_rows)[:limit]
    if bridge_row is None or bridge_score < min_score:
        return rows
    if bridge_score >= top_book_score or not rows:
        return ([bridge_row] + rows)[:limit]
    return (rows[: max(limit - 1, 0)] + [bridge_row])[:limit]
