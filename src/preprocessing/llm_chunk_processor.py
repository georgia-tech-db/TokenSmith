"""
src/preprocessing/llm_chunk_processor.py

LLM-based chunk post-processing for TokenSmith.
Called by index_builder.py when the corresponding config flags are set.

Stage 2 — reorganize():
    Sliding-window LLM pass that merges semantically inseparable adjacent
    chunks and splits chunks covering unrelated topics.  Text is NEVER
    rewritten — only chunk boundaries are adjusted.

Stage 3 — resolve_coreferences():
    Per-chunk pass that replaces vague pronouns and ambiguous nominals
    ("it", "the system", "the protocol") with their explicit referents
    so each chunk is maximally self-contained for retrieval.

Both stages are crash-safe via per-call JSONL logging in tmp/.
On clean completion each stage writes its output pkl to tmp/ AND copies
permanent records to index/log_files/.
"""
from __future__ import annotations

import copy
import json
import os
import pathlib
import pickle
import re
import shutil
import time
from typing import Dict, List, Optional, Tuple

import requests

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Models are hardcoded here — no need to surface them in config.yaml.
REORG_MODEL  = "google/gemini-2.0-flash-001"
COREF_MODEL  = "anthropic/claude-haiku-4-5"

REORG_WINDOW_SIZE = 5
REORG_MAX_TOKENS  = 8000
COREF_MAX_TOKENS  = 2000

MAX_RETRIES      = 3
RETRY_DELAY_BASE = 8    # seconds; multiplied by attempt number on each retry
API_THROTTLE     = 1.0  # seconds between successful API calls

# Sentence verification: ignore fragments shorter than this many characters.
# Short strings (isolated numbers, section markers) produce too many false
# positives during substring matching.
MIN_SENTENCE_LEN = 20

# Coreference validation thresholds
MAX_WORD_COUNT_CHANGE_RATIO = 0.15  # >15% change → fall back to original
MAX_SENTENCE_COUNT_CHANGE   = 1     # adding/removing >1 sentence → fall back


# ─────────────────────────────────────────────────────────────────────────────
# HTTP helper
# ─────────────────────────────────────────────────────────────────────────────

def _call_openrouter(
    messages: List[Dict],
    model: str,
    max_tokens: int,
) -> Optional[str]:
    """
    POST to OpenRouter's chat completion endpoint.
    Returns the response text string or None after all retries are exhausted.
    Raises EnvironmentError immediately if OPENROUTER_API_KEY is not set.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "OPENROUTER_API_KEY environment variable is not set. "
            "Export it before running the indexer with LLM stages enabled."
        )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(
                OPENROUTER_URL,
                headers=headers,
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            print(f"    [LLM] Attempt {attempt}/{MAX_RETRIES} failed: {exc}")
            if attempt < MAX_RETRIES:
                delay = RETRY_DELAY_BASE * attempt
                print(f"    [LLM] Retrying in {delay}s …")
                time.sleep(delay)
    return None


def _parse_json(raw: str) -> Optional[Dict]:
    """Strip optional markdown fences and parse JSON. Returns None on failure."""
    cleaned = re.sub(
        r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE
    ).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Last-resort: find the outermost { … } object
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Shared text helpers
# ─────────────────────────────────────────────────────────────────────────────

def _normalise(text: str) -> str:
    """Collapse whitespace runs to a single space and strip."""
    return re.sub(r"\s+", " ", text).strip()


def _sentences(text: str) -> List[str]:
    """
    Rough sentence splitter.
    Returns only non-trivial sentences (>= MIN_SENTENCE_LEN characters) to
    reduce noise during substring verification.
    """
    parts = re.split(r"(?<=[.?!])\s+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) >= MIN_SENTENCE_LEN]


def _verify_chunk(output_text: str, source_texts: List[str]) -> List[str]:
    """
    Returns the list of sentences in output_text that are not found as
    a substring (after whitespace normalisation) in any of source_texts.
    An empty return list means the chunk passed verification.
    """
    norm_sources = [_normalise(s) for s in source_texts]
    failures: List[str] = []
    for sentence in _sentences(output_text):
        if not any(_normalise(sentence) in src for src in norm_sources):
            failures.append(sentence)
    return failures


# ─────────────────────────────────────────────────────────────────────────────
# Shared metadata helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_reorg_chunk_metadata(
    chunk_text: str,
    new_origins: List[int],
    original_metadata: List[Dict],
    new_chunk_id: int,
) -> Dict:
    """
    Build a metadata dict for one reorganized output chunk.

    Provenance rules
    ----------------
    filename, mode, section, section_path  →  taken from the FIRST contributing
                                               original chunk (new_origins[0])
    page_numbers                           →  UNION across all contributing
                                               original chunks
    char_len, word_len, text_preview       →  recalculated from the new text
    chunk_id                               →  new sequential id
    """
    first = original_metadata[new_origins[0]] if new_origins else {}

    all_pages: set = set()
    for orig_idx in new_origins:
        if orig_idx < len(original_metadata):
            all_pages.update(original_metadata[orig_idx].get("page_numbers", []))

    return {
        "filename":               first.get("filename", ""),
        "mode":                   first.get("mode", ""),
        "llm_based_reorg":        True,
        "llm_based_coref_res":    False,   # will be flipped by coref stage if it runs
        "char_len":               len(chunk_text),
        "word_len":               len(chunk_text.split()),
        "section":                first.get("section", ""),
        "section_path":           first.get("section_path", ""),
        "text_preview":           chunk_text[:100],
        "page_numbers":           sorted(list(all_pages)),
        "chunk_id":               new_chunk_id,
        "reorg_source_chunk_ids": new_origins,   # provenance audit trail
    }


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 2 — CHUNK REORGANIZER
# ═════════════════════════════════════════════════════════════════════════════

_REORG_SYSTEM = (
    "You are a precise text-chunk reorganizer for a Retrieval-Augmented Generation "
    "system. Your only job is to adjust chunk boundaries. You must NEVER rewrite, "
    "paraphrase, add, or remove any text."
)


def _build_reorg_prompt(window_chunks: List[str], has_carry_over: bool) -> str:
    carry_note = ""
    if has_carry_over:
        carry_note = (
            "\nSPECIAL NOTE — CHUNK 0 is a CARRY-OVER from the previous batch. "
            "Content from later chunks may be merged INTO it if they continue the "
            "same thought. It can also be split if clearly necessary. "
            "Otherwise treat it like any other chunk.\n"
        )

    blocks: List[str] = []
    for i, text in enumerate(window_chunks):
        label = f"CHUNK {i}" + (" ← CARRY-OVER" if i == 0 and has_carry_over else "")
        blocks.append(f"=== {label} ===\n{text}\n=== END {label} ===")

    return f"""You are given {len(window_chunks)} text chunks from a database textbook.
Reorganize them into better semantic units for retrieval.
{carry_note}
━━━ ABSOLUTE RULES — any violation is a critical failure ━━━
1. Do NOT change a single word, character, or punctuation mark in any sentence.
   Every sentence in your output MUST be a verbatim copy from one of the input chunks.
2. MERGE two or more ADJACENT chunks only when they form one continuous explanation,
   example, analogy, or algorithm that loses meaning when read separately.
3. SPLIT one chunk only when it contains clearly UNRELATED topics — an abrupt
   topic change with no logical connection between the two halves.
4. If a chunk is already coherent and self-contained, mark it "unchanged".
5. Every input index (0 through {len(window_chunks) - 1}) MUST appear in at least
   one output chunk's source_indices list. No input chunk may be silently dropped.

━━━ OUTPUT FORMAT ━━━
Respond with ONLY a valid JSON object — no markdown fences, no text outside the JSON:

{{
  "chunks": [
    {{
      "text":           "<verbatim sentences drawn exactly from the input>",
      "status":         "unchanged" | "merged" | "split",
      "source_indices": [<0-based indices of the input chunks whose text appears here>],
      "reasoning":      "<one sentence>"
    }}
  ]
}}

━━━ INPUT CHUNKS ━━━

{chr(10).join(blocks)}
"""


def _process_reorg_window(
    window_chunks: List[str],
    window_origins: List[List[int]],
    has_carry_over: bool,
    window_num: int,
) -> Tuple[List[Dict], List[str]]:
    """
    Call the LLM for one reorganization window.

    Each output chunk dict gets an additional 'new_origins' field that
    tracks which ORIGINAL (stage-1) chunk indices contributed to it.
    This is computed by flattening window_origins[source_indices].

    Returns (output_chunk_dicts, all_verification_failure_sentences).
    Falls back to a verbatim pass-through on LLM failure or JSON error.
    """
    prompt = _build_reorg_prompt(window_chunks, has_carry_over)
    raw = _call_openrouter(
        [{"role": "system", "content": _REORG_SYSTEM},
         {"role": "user",   "content": prompt}],
        REORG_MODEL,
        REORG_MAX_TOKENS,
    )

    if raw is None:
        print(f"  [W{window_num}] LLM failed all retries → pass-through")
        return _reorg_passthrough(window_chunks, window_origins), []

    parsed = _parse_json(raw)
    if parsed is None or "chunks" not in parsed:
        print(f"  [W{window_num}] JSON parse failed → pass-through")
        print(f"             Raw (first 300 chars): {(raw or '')[:300]}")
        return _reorg_passthrough(window_chunks, window_origins), []

    output_chunks: List[Dict] = parsed["chunks"]

    # ── Ensure every input index is accounted for ─────────────────────────────
    covered = {idx for c in output_chunks for idx in c.get("source_indices", [])}
    missing = set(range(len(window_chunks))) - covered
    if missing:
        print(f"  [W{window_num}] ⚠  Indices {sorted(missing)} missing from output — inserting as-is")
        for idx in sorted(missing):
            output_chunks.append({
                "text":           window_chunks[idx],
                "status":         "unchanged",
                "source_indices": [idx],
                "reasoning":      "auto-inserted: not covered by LLM output",
            })

    # ── Attach new_origins and run sentence verification ──────────────────────
    all_failures: List[str] = []
    for chunk_data in output_chunks:
        src_idxs = chunk_data.get("source_indices", [])

        # Flatten window_origins for the referenced input chunks, dedup, preserve order
        flat: List[int] = []
        seen: set = set()
        for idx in src_idxs:
            if idx < len(window_origins):
                for o in window_origins[idx]:
                    if o not in seen:
                        seen.add(o)
                        flat.append(o)
        chunk_data["new_origins"] = flat

        # Sentence-level verification against the claimed source chunks
        relevant = [window_chunks[i] for i in src_idxs if i < len(window_chunks)] or window_chunks
        failures = _verify_chunk(chunk_data["text"], relevant)
        chunk_data["_verify_failures"] = failures
        if failures:
            all_failures.extend(failures)
            print(f"  [W{window_num}] ⚠  {len(failures)} verification failure(s):")
            for s in failures[:2]:
                print(f"         ↳ '{s[:90]}'")
            if len(failures) > 2:
                print(f"         ↳  … and {len(failures) - 2} more")

    return output_chunks, all_failures


def _reorg_passthrough(
    window_chunks: List[str],
    window_origins: List[List[int]],
) -> List[Dict]:
    """Return every input chunk unchanged, preserving its origins."""
    return [
        {
            "text":             c,
            "status":           "unchanged",
            "source_indices":   [i],
            "reasoning":        "pass-through (LLM failure or JSON error)",
            "new_origins":      window_origins[i],
            "_verify_failures": [],
        }
        for i, c in enumerate(window_chunks)
    ]


def _dedup_adjacent(
    chunks: List[str],
    origins: List[List[int]],
) -> Tuple[List[str], List[List[int]]]:
    """
    Remove near-identical ADJACENT chunk pairs that arise when the carry-over
    is emitted unchanged in both the outgoing and incoming window.

    Strategy:
      exact match     → drop second
      curr ⊂ prev     → drop curr (prev already contains it)
      prev ⊂ curr     → replace prev with curr (curr is the longer version)
    """
    if not chunks:
        return chunks, origins
    out_c: List[str]        = [chunks[0]]
    out_o: List[List[int]]  = [origins[0]]
    for chunk, orig in zip(chunks[1:], origins[1:]):
        prev_n = _normalise(out_c[-1])
        curr_n = _normalise(chunk)
        if curr_n == prev_n:
            continue
        if curr_n in prev_n:
            continue
        if prev_n in curr_n:
            out_c[-1] = chunk
            out_o[-1] = orig
            continue
        out_c.append(chunk)
        out_o.append(orig)
    return out_c, out_o


def reorganize(
    chunks: List[str],
    metadata: List[Dict],
    tmp_dir: pathlib.Path,
    log_files_dir: pathlib.Path,
    prefix: str,
) -> Tuple[List[str], List[Dict]]:
    """
    Stage 2: LLM-based sliding-window chunk reorganization.

    Crash-safe via a per-window JSONL in tmp_dir.  On clean completion the
    output pkl files are written to tmp_dir AND copied to log_files_dir.

    Parameters
    ----------
    chunks        : clean chunk texts from stage 1 (no heading prefix)
    metadata      : parallel metadata list from stage 1
    tmp_dir       : pathlib.Path to the tmp/ directory
    log_files_dir : pathlib.Path to index/log_files/
    prefix        : index_prefix string used to name log files

    Returns
    -------
    (new_chunks, new_metadata)  — parallel lists of reorganized chunks and
                                  their updated metadata
    """
    jsonl_path         = tmp_dir / "stage2_reorg_windows.jsonl"
    stage2_chunks_path = tmp_dir / "stage2_reorg_chunks.pkl"
    stage2_meta_path   = tmp_dir / "stage2_reorg_meta.pkl"

    tmp_dir.mkdir(parents=True, exist_ok=True)
    log_files_dir.mkdir(parents=True, exist_ok=True)

    total = len(chunks)
    print(f"\n{'='*60}")
    print(f"[STAGE 2 — REORG] {total} input chunks")
    print(f"  model       : {REORG_MODEL}")
    print(f"  window size : {REORG_WINDOW_SIZE}")
    print(f"  JSONL log   : {jsonl_path}")
    print(f"{'='*60}")

    # ── Load existing JSONL for crash recovery ────────────────────────────────
    existing_log: List[Dict] = []
    if jsonl_path.exists():
        with open(jsonl_path) as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    existing_log.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"  [WARN] Malformed JSONL on line {lineno} — skipping")
        print(f"  [RESUME] Found {len(existing_log)} completed windows — resuming")

    # ── Reconstruct in-memory state from the log ──────────────────────────────
    emitted_chunks:  List[str]       = []
    emitted_origins: List[List[int]] = []
    carry_over_text:    Optional[str]       = None
    carry_over_origins: Optional[List[int]] = None
    orig_idx: int = 0

    for entry in existing_log:
        if entry.get("status") != "success":
            print(f"  [RESUME] Window {entry.get('window_num')} not successful — "
                  f"stopping resume here")
            # Truncate to only the good entries processed so far
            break
        out = entry["output_chunks"]
        if entry["is_last_window"]:
            emitted_chunks.extend(c["text"]        for c in out)
            emitted_origins.extend(c["new_origins"] for c in out)
            carry_over_text    = None
            carry_over_origins = None
        else:
            emitted_chunks.extend(c["text"]        for c in out[:-1])
            emitted_origins.extend(c["new_origins"] for c in out[:-1])
            carry_over_text    = out[-1]["text"]
            carry_over_origins = out[-1]["new_origins"]
        orig_idx = entry["orig_new_end"]

    print(f"  [INFO] Starting from orig_idx={orig_idx} / {total}")
    window_num = len(existing_log)

    # ── Sliding window ────────────────────────────────────────────────────────
    while orig_idx < total:
        has_co    = carry_over_text is not None
        new_count = min(REORG_WINDOW_SIZE - (1 if has_co else 0), total - orig_idx)
        is_last   = (orig_idx + new_count) >= total

        # Build the window input lists (text and parallel origins)
        window_chunks:  List[str]       = []
        window_origins: List[List[int]] = []

        if has_co:
            window_chunks.append(carry_over_text)
            window_origins.append(carry_over_origins)

        for i in range(new_count):
            window_chunks.append(chunks[orig_idx + i])
            # Origins for an untouched stage-1 chunk is just its own index
            window_origins.append([orig_idx + i])

        print(
            f"\n  [W{window_num}] orig=[{orig_idx}:{orig_idx + new_count}] "
            f"win_len={len(window_chunks)} carry_over={'yes' if has_co else 'no'} "
            f"last={is_last}"
        )

        output_chunks, failures = _process_reorg_window(
            window_chunks, window_origins, has_co, window_num
        )

        # ── Append to JSONL ───────────────────────────────────────────────────
        log_entry: Dict = {
            "window_num":            window_num,
            "orig_new_start":        orig_idx,
            "orig_new_end":          orig_idx + new_count,
            "has_carry_over":        has_co,
            "is_last_window":        is_last,
            "window_len":            len(window_chunks),
            "output_count":          len(output_chunks),
            "output_chunks":         output_chunks,
            "total_verify_failures": len(failures),
            "status":                "success",
        }
        with open(jsonl_path, "a") as fh:
            fh.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        # ── Emit (hold last chunk as carry-over for non-final windows) ────────
        if is_last:
            emitted_chunks.extend(c["text"]        for c in output_chunks)
            emitted_origins.extend(c["new_origins"] for c in output_chunks)
            carry_over_text    = None
            carry_over_origins = None
        else:
            emitted_chunks.extend(c["text"]        for c in output_chunks[:-1])
            emitted_origins.extend(c["new_origins"] for c in output_chunks[:-1])
            carry_over_text    = output_chunks[-1]["text"]
            carry_over_origins = output_chunks[-1]["new_origins"]

        orig_idx  += new_count
        window_num += 1
        time.sleep(API_THROTTLE)

    # ── Deduplication ─────────────────────────────────────────────────────────
    final_chunks, final_origins = _dedup_adjacent(emitted_chunks, emitted_origins)
    removed = len(emitted_chunks) - len(final_chunks)
    if removed:
        print(f"\n  [DEDUP] Removed {removed} near-duplicate carry-over chunk(s)")

    print(f"\n  [DONE] {total} → {len(final_chunks)} chunks ({window_num} windows)")

    # ── Build new metadata ────────────────────────────────────────────────────
    new_metadata: List[Dict] = [
        _build_reorg_chunk_metadata(chunk, origins, metadata, new_id)
        for new_id, (chunk, origins) in enumerate(zip(final_chunks, final_origins))
    ]

    # ── Persist to tmp ────────────────────────────────────────────────────────
    with open(stage2_chunks_path, "wb") as f:
        pickle.dump(final_chunks, f)
    with open(stage2_meta_path, "wb") as f:
        pickle.dump(new_metadata, f)
    print(f"  [TMP] Saved stage2 pkl files to {tmp_dir}/")

    # ── Copy to permanent log_files ───────────────────────────────────────────
    log_files_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(stage2_chunks_path, log_files_dir / f"{prefix}_stage2_reorg_chunks.pkl")
    shutil.copy2(stage2_meta_path,   log_files_dir / f"{prefix}_stage2_reorg_meta.pkl")
    shutil.copy2(jsonl_path,         log_files_dir / f"{prefix}_stage2_reorg_windows.jsonl")
    print(f"  [LOG] Copied stage2 records to {log_files_dir}/")

    return final_chunks, new_metadata


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 3 — COREFERENCE RESOLVER
# ═════════════════════════════════════════════════════════════════════════════

_COREF_SYSTEM = (
    "You are an expert in natural-language anaphora resolution for technical text. "
    "Replace vague references with their specific referents — nothing else. "
    "Be conservative: when in doubt, leave the text unchanged."
)

_NO_PREV = "(no previous chunk — this is the first chunk in the document)"


def _build_coref_prompt(current_chunk: str, prev_chunk: Optional[str]) -> str:
    prev_text = prev_chunk if prev_chunk else _NO_PREV
    return f"""You are resolving anaphoric references inside a text chunk from a database textbook.

━━━ CONTEXT (previous chunk — read-only, do NOT include in output) ━━━
{prev_text}
━━━ END CONTEXT ━━━

━━━ CURRENT CHUNK TO RESOLVE ━━━
{current_chunk}
━━━ END CURRENT CHUNK ━━━

Replace vague or ambiguous references in the CURRENT CHUNK ONLY with their
specific, unambiguous referents so the chunk can be understood in isolation.

━━━ RULES ━━━
1. Only replace a reference when you are HIGHLY CONFIDENT (>95%) of its referent.
   If there is any doubt, leave it unchanged.
2. Do NOT add, remove, reorder, or rephrase any sentence.
3. Do NOT change anything except the specific vague phrases being resolved.
4. Common targets:
     pronouns       : "it", "this", "they", "them", "these", "those", "its"
     vague nominals : "the system", "the protocol", "the method", "the algorithm",
                      "the technique", "the approach", "the scheme", "the mechanism",
                      "the structure", "the format", "the process", "the operation"
5. Do NOT resolve expletive "it" (e.g. "It is important that…", "It follows that…").
6. Do NOT replace already-specific terms (e.g. "the B+ tree", "SQL", "ARIES").
7. If no changes are needed return the original text unchanged with an empty changes list.

━━━ OUTPUT FORMAT ━━━
Respond with ONLY a valid JSON object — no markdown fences, no prose outside the JSON:

{{
  "resolved_text": "<chunk text with resolved references>",
  "changes": [
    {{
      "original_phrase": "<vague phrase as it appears in the text>",
      "replacement":     "<specific term used>",
      "confidence":      "high" | "medium",
      "note":            "<brief justification>"
    }}
  ]
}}
"""


def _validate_coref_output(original: str, resolved: str) -> Tuple[bool, str]:
    """
    Sanity-check the resolved text against the original.
    Returns (is_valid, reason_string).
    """
    orig_wc = len(original.split())
    res_wc  = len(resolved.split())
    if orig_wc > 0:
        ratio = abs(res_wc - orig_wc) / orig_wc
        if ratio > MAX_WORD_COUNT_CHANGE_RATIO:
            return False, (
                f"word count changed by {ratio:.1%} ({orig_wc}→{res_wc}), "
                f"exceeds {MAX_WORD_COUNT_CHANGE_RATIO:.0%} threshold"
            )

    orig_sc = len(re.split(r"(?<=[.?!])\s+", original.strip()))
    res_sc  = len(re.split(r"(?<=[.?!])\s+", resolved.strip()))
    sc_diff = abs(res_sc - orig_sc)
    if sc_diff > MAX_SENTENCE_COUNT_CHANGE:
        return False, (
            f"sentence count changed by {sc_diff} ({orig_sc}→{res_sc}), "
            f"exceeds threshold of {MAX_SENTENCE_COUNT_CHANGE}"
        )

    return True, ("no changes" if _normalise(resolved) == _normalise(original) else "ok")


def _build_coref_metadata(original_meta: Dict, resolved_text: str) -> Dict:
    """
    Coref resolution doesn't change chunk count or boundaries.
    Only size-based fields and the provenance flag change.
    """
    updated = copy.deepcopy(original_meta)
    updated.update({
        "char_len":            len(resolved_text),
        "word_len":            len(resolved_text.split()),
        "text_preview":        resolved_text[:100],
        "llm_based_coref_res": True,
    })
    return updated


def resolve_coreferences(
    chunks: List[str],
    metadata: List[Dict],
    tmp_dir: pathlib.Path,
    log_files_dir: pathlib.Path,
    prefix: str,
) -> Tuple[List[str], List[Dict]]:
    """
    Stage 3: LLM-based coreference/anaphora resolution.

    Each chunk is processed with the RESOLVED previous chunk as context (not
    the original), so if chunk N resolved "it" → "SQL", chunk N+1's LLM call
    correctly sees "SQL" as the antecedent.

    Crash-safe via a per-chunk JSONL in tmp_dir.  On clean completion the
    output pkl files are written to tmp_dir AND copied to log_files_dir.

    Parameters
    ----------
    chunks        : clean chunk texts (output of stage 2, or stage 1 if reorg disabled)
    metadata      : parallel metadata list
    tmp_dir       : pathlib.Path to the tmp/ directory
    log_files_dir : pathlib.Path to index/log_files/
    prefix        : index_prefix string used to name log files

    Returns
    -------
    (resolved_chunks, updated_metadata)
    """
    jsonl_path         = tmp_dir / "stage3_coref_chunks.jsonl"
    stage3_chunks_path = tmp_dir / "stage3_coref_chunks.pkl"
    stage3_meta_path   = tmp_dir / "stage3_coref_meta.pkl"

    tmp_dir.mkdir(parents=True, exist_ok=True)
    log_files_dir.mkdir(parents=True, exist_ok=True)

    total = len(chunks)
    print(f"\n{'='*60}")
    print(f"[STAGE 3 — COREF] {total} input chunks")
    print(f"  model     : {COREF_MODEL}")
    print(f"  JSONL log : {jsonl_path}")
    print(f"{'='*60}")

    # ── Load existing JSONL for crash recovery ────────────────────────────────
    existing_log: List[Dict] = []
    if jsonl_path.exists():
        with open(jsonl_path) as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    existing_log.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"  [WARN] Malformed JSONL on line {lineno} — skipping")

    # Validate log continuity: stop at the first non-success entry
    valid_log: List[Dict] = []
    for entry in existing_log:
        if entry.get("status") != "success":
            print(f"  [RESUME] Entry {entry.get('chunk_idx')} not successful — "
                  f"stopping resume here")
            break
        valid_log.append(entry)

    completed = len(valid_log)
    if completed:
        print(f"  [RESUME] {completed} chunks already resolved — resuming from index {completed}")

    # Reconstruct resolved list from the valid log entries
    resolved_chunks: List[str] = [e["resolved_text"] for e in valid_log]

    # ── Process remaining chunks ──────────────────────────────────────────────
    for idx in range(completed, total):
        chunk = chunks[idx]
        # Use the already-resolved previous chunk as context — not the original
        prev  = resolved_chunks[-1] if resolved_chunks else None

        if idx > 0 and idx % 100 == 0:
            print(f"  [PROGRESS] {idx}/{total} chunks processed …")

        # ── LLM call ─────────────────────────────────────────────────────────
        prompt = _build_coref_prompt(chunk, prev)
        raw = _call_openrouter(
            [{"role": "system", "content": _COREF_SYSTEM},
             {"role": "user",   "content": prompt}],
            COREF_MODEL,
            COREF_MAX_TOKENS,
        )

        used_fallback = False
        changes: List[Dict] = []
        resolved_text = chunk  # default: keep original

        if raw is None:
            print(f"  [C{idx}] LLM failed → keeping original")
            used_fallback = True
        else:
            parsed = _parse_json(raw)
            if parsed is None or "resolved_text" not in parsed:
                print(f"  [C{idx}] JSON parse failed → keeping original")
                used_fallback = True
            else:
                candidate = parsed.get("resolved_text", chunk).strip()
                changes   = parsed.get("changes", [])
                is_valid, reason = _validate_coref_output(chunk, candidate)
                if not is_valid:
                    print(f"  [C{idx}] Validation failed ({reason}) → keeping original")
                    used_fallback = True
                else:
                    resolved_text = candidate
                    if changes:
                        print(f"  [C{idx}] {len(changes)} reference(s) resolved:")
                        for ch in changes[:4]:
                            print(
                                f"         '{ch.get('original_phrase','?')}' → "
                                f"'{ch.get('replacement','?')}' "
                                f"[{ch.get('confidence','?')}]"
                            )
                        if len(changes) > 4:
                            print(f"         … and {len(changes) - 4} more")

        # ── JSONL log ─────────────────────────────────────────────────────────
        log_entry: Dict = {
            "chunk_idx":     idx,
            "original_text": chunk,
            "resolved_text": resolved_text,
            "changes":       changes,
            "num_changes":   len(changes),
            "used_fallback": used_fallback,
            "status":        "success",
        }
        with open(jsonl_path, "a") as fh:
            fh.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        resolved_chunks.append(resolved_text)
        time.sleep(API_THROTTLE)

    # ── Stats ─────────────────────────────────────────────────────────────────
    full_log = valid_log + [
        json.loads(line)
        for line in open(jsonl_path).readlines()[completed:]
        if line.strip()
    ]
    total_changes  = sum(e.get("num_changes", 0) for e in full_log)
    chunks_changed = sum(1 for e in full_log if e.get("num_changes", 0) > 0)
    fallbacks      = sum(1 for e in full_log if e.get("used_fallback", False))

    print(f"\n  [DONE] {total} chunks processed")
    print(f"         Chunks with ≥1 change : {chunks_changed}")
    print(f"         Total changes          : {total_changes}")
    print(f"         Fallbacks to original  : {fallbacks}")

    # ── Build updated metadata ────────────────────────────────────────────────
    new_metadata: List[Dict] = [
        _build_coref_metadata(meta, resolved)
        for meta, resolved in zip(metadata, resolved_chunks)
    ]

    # ── Persist to tmp ────────────────────────────────────────────────────────
    with open(stage3_chunks_path, "wb") as f:
        pickle.dump(resolved_chunks, f)
    with open(stage3_meta_path, "wb") as f:
        pickle.dump(new_metadata, f)
    print(f"  [TMP] Saved stage3 pkl files to {tmp_dir}/")

    # ── Copy to permanent log_files ───────────────────────────────────────────
    log_files_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(stage3_chunks_path, log_files_dir / f"{prefix}_stage3_coref_chunks.pkl")
    shutil.copy2(stage3_meta_path,   log_files_dir / f"{prefix}_stage3_coref_meta.pkl")
    shutil.copy2(jsonl_path,         log_files_dir / f"{prefix}_stage3_coref_chunks.jsonl")
    print(f"  [LOG] Copied stage3 records to {log_files_dir}/")

    return resolved_chunks, new_metadata