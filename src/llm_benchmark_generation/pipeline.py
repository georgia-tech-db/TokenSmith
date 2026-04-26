"""
src/llm_benchmark_generation/pipeline.py

Per-QAC agentic refinement pipeline (Steps 2-6).

Step 2  — Self-critique       (no pages, cheap)
Step 3  — Standard refine     (no pages, only if issues found)
Step 3b — Fallback refine     (targeted pages, only if chunks_insufficient)
Step 4  — First verification  (two models, no pages)
Step 5  — Second refine       (no pages, only if Step 4 flagged anything)
Step 6  — Final verification  (two models, no pages)

Pages are sent at most once per window (Step 1 in generator.py) plus a
targeted slice in Step 3b if the fallback fires.

All prompt/response pairs are stored in the per-QAC log for full auditability.
"""

from __future__ import annotations

import time
from copy import deepcopy
from typing import Optional

from llm_client   import call_llm, parse_json_response
from prompts      import (
    CRITIQUE_SYSTEM, build_critique_prompt,
    REFINE_SYSTEM,   build_refine_prompt, build_fallback_prompt,
    VERIFY_SYSTEM,   build_verify_prompt, VERIFY_CRITERIA,
)
from verification import verify_gold_chunks
from markdown_utils import extract_pages


# ─────────────────────────────────────────────────────────────────────────────
# Log entry builder
# ─────────────────────────────────────────────────────────────────────────────

def _stage_entry(
    stage:    str,
    model:    str,
    prompt:   str,
    response: Optional[str],
    usage:    Optional[dict],
    parsed:   Optional[dict] = None,
) -> dict:
    import time as _time
    return {
        "stage":     stage,
        "model":     model,
        "prompt":    prompt,
        "response":  response,
        "usage":     usage,
        "parsed":    parsed,
        "timestamp": _time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Targeted page extraction for Step 3b fallback
# ─────────────────────────────────────────────────────────────────────────────

def _get_targeted_pages(
    record:       dict,
    full_md:      str,
    offsets:      dict[int, int],
    window_start: int,
    window_end:   int,
) -> str:
    """
    Find which pages the existing gold chunks appear on and return those
    pages +/- 1 as context for the fallback refine step.
    Falls back to the first 3 pages of the window on detection failure.
    """
    from verification import check_chunk_in_pages, normalise_ws, clean_md_for_verification

    chunk_pages: set[int] = set()
    for chunk in record.get("gold_chunks", []):
        for page_num in range(window_start, window_end + 1):
            if page_num not in offsets:
                continue
            try:
                page_text = extract_pages(full_md, offsets, page_num, page_num)
                if check_chunk_in_pages(chunk, page_text):
                    chunk_pages.add(page_num)
                    break
            except ValueError:
                continue

    if not chunk_pages:
        t_start = window_start
        t_end   = min(window_start + 2, window_end)
    else:
        t_start = max(window_start, min(chunk_pages) - 1)
        t_end   = min(window_end,   max(chunk_pages) + 1)

    try:
        return extract_pages(full_md, offsets, t_start, t_end)
    except ValueError:
        return extract_pages(full_md, offsets, window_start,
                             min(window_start + 2, window_end))


# ─────────────────────────────────────────────────────────────────────────────
# Verifier runner
# ─────────────────────────────────────────────────────────────────────────────

def _run_verifiers(
    record:         dict,
    verifier_1:     str,
    verifier_2:     str,
    api_key:        str,
    cfg:            dict,
    stage_label:    str,
) -> tuple[dict, bool, list[str]]:
    """
    Run both verifier models on one QAC record.
    Returns (results_dict, flagged, flag_reasons).
    results_dict has keys "verifier_1" and "verifier_2".
    """
    prompt   = build_verify_prompt(record)
    messages = [
        {"role": "system", "content": VERIFY_SYSTEM},
        {"role": "user",   "content": prompt},
    ]
    results: dict[str, dict] = {}

    for key, model_id in [("verifier_1", verifier_1), ("verifier_2", verifier_2)]:
        print(f"      [{stage_label}] {key} ({model_id}) ...", end=" ", flush=True)
        raw, usage = call_llm(
            messages,
            model=model_id,
            api_key=api_key,
            max_tokens=cfg.get("max_tokens_verify", 1500),
            connect_timeout=cfg.get("http_connect_timeout", 10),
            read_timeout=cfg.get("http_read_timeout", 180),
            retries=cfg.get("max_retries", 3),
            retry_delay=cfg.get("retry_delay_base", 10),
        )
        if raw is None:
            print("FAILED (LLM error)")
            results[key] = {
                "error": "LLM call failed",
                "prompt": prompt, "response": None, "usage": None,
            }
            continue

        parsed = parse_json_response(raw)
        if not isinstance(parsed, dict):
            print("FAILED (JSON parse)")
            results[key] = {
                "error": "JSON parse failed",
                "prompt": prompt, "response": raw, "usage": usage,
            }
            continue

        verdicts = [parsed.get(c, "uncertain") for c in VERIFY_CRITERIA]
        overall  = "passed" if all(v == "passed" for v in verdicts) else "failed"
        parsed["overall"] = overall
        results[key] = {
            "prompt": prompt, "response": raw, "usage": usage, "parsed": parsed,
        }
        print(overall.upper())
        time.sleep(0.5)

    # Collect flag reasons
    flagged      = False
    flag_reasons = []
    for key, result in results.items():
        if "error" in result:
            flagged = True
            flag_reasons.append(f"{key}: {result['error']}")
            continue
        parsed = result.get("parsed", {})
        for c in VERIFY_CRITERIA:
            verdict = parsed.get(c, "uncertain")
            if verdict in ("failed", "uncertain"):
                flagged = True
                note    = parsed.get("notes", {}).get(c, "")
                flag_reasons.append(f"{key}.{c}={verdict}: {note}")

    return results, flagged, flag_reasons


# ─────────────────────────────────────────────────────────────────────────────
# Main per-QAC pipeline function
# ─────────────────────────────────────────────────────────────────────────────

def run_qac_pipeline(
    qac:          dict,
    qac_index:    int,
    pages_text:   str,
    full_md:      str,
    offsets:      dict[int, int],
    window_start: int,
    window_end:   int,
    generation_model: str,
    verifier_1:   str,
    verifier_2:   str,
    api_key:      str,
    cfg:          dict,
) -> tuple[dict, dict]:
    """
    Run Steps 2-6 for one QAC pair.

    The substring check (Step A) always runs but never blocks the pipeline.
    If it fails the QAC is forced to manual_review regardless of what the
    refinement/verification steps conclude.

    Returns (final_qac, qac_log).
    """
    qac_log = {
        "qac_index": qac_index,
        "initial":   qac,
        "stages":    {},
        "final_qac": None,
        "status":    None,
    }
    current_qac  = deepcopy(qac)
    did_fallback = False

    print(
        f"\n  QAC {qac_index} [{current_qac.get('difficulty','?').upper()}] "
        f"{current_qac.get('question','')[:65]} ..."
    )

    # ── Substring check ───────────────────────────────────────────────────────
    substr_result    = verify_gold_chunks(current_qac, pages_text)
    qac_log["stages"]["substring_check"] = substr_result
    substring_failed = not substr_result["passed"]

    if substring_failed:
        print(
            f"    [SUBSTRING] FAILED — {len(substr_result['failures'])} chunk(s) "
            f"not found verbatim — pipeline continues but will force manual_review"
        )
        for fail in substr_result["failures"]:
            print(f"       x '{fail[:80]}'")
    else:
        print(f"    [SUBSTRING] passed ({len(current_qac.get('gold_chunks', []))} chunks)")

    # ── Step 2: Self-critique ─────────────────────────────────────────────────
    print("    [Step 2] Self-critique ...", end=" ", flush=True)
    crit_prompt = build_critique_prompt(current_qac)
    raw_c, usage_c = call_llm(
        [{"role": "system", "content": CRITIQUE_SYSTEM},
         {"role": "user",   "content": crit_prompt}],
        model=generation_model, api_key=api_key,
        max_tokens=cfg.get("max_tokens_critique", 2000),
        connect_timeout=cfg.get("http_connect_timeout", 10),
        read_timeout=cfg.get("http_read_timeout", 180),
        retries=cfg.get("max_retries", 3),
        retry_delay=cfg.get("retry_delay_base", 10),
    )
    crit_parsed = parse_json_response(raw_c) if raw_c else None
    if not isinstance(crit_parsed, dict):
        print(f"  [WARN] Critique returned unexpected type — skipping refine")
        crit_parsed = None

    qac_log["stages"]["critique"] = _stage_entry(
        "critique", generation_model, crit_prompt, raw_c, usage_c, crit_parsed
    )

    has_issues    = False
    needs_fallback = False
    if crit_parsed:
        has_issues = bool(
            crit_parsed.get("phrasing_issue")
            or crit_parsed.get("terminology_mismatch")
            or crit_parsed.get("chunks_redundant")
            or crit_parsed.get("rubric_issue")
            or crit_parsed.get("mock_answer_issue")
        )
        needs_fallback = bool(crit_parsed.get("chunks_insufficient"))

    # ── Step 3b: Fallback ─────────────────────────────────────────────────────
    if needs_fallback:
        print("    [Step 3b] Chunks insufficient — fetching targeted pages ...")
        targeted = _get_targeted_pages(
            current_qac, full_md, offsets, window_start, window_end
        )
        fb_prompt = build_fallback_prompt(current_qac, crit_parsed, targeted)
        raw_fb, usage_fb = call_llm(
            [{"role": "system", "content": REFINE_SYSTEM},
             {"role": "user",   "content": fb_prompt}],
            model=generation_model, api_key=api_key,
            max_tokens=cfg.get("max_tokens_refine", 4000),
            connect_timeout=cfg.get("http_connect_timeout", 10),
            read_timeout=cfg.get("http_read_timeout", 180),
            retries=cfg.get("max_retries", 3),
            retry_delay=cfg.get("retry_delay_base", 10),
        )
        fb_parsed = parse_json_response(raw_fb) if raw_fb else None
        if not isinstance(fb_parsed, dict):
            fb_parsed = None
        qac_log["stages"]["fallback_refine"] = _stage_entry(
            "fallback_refine", generation_model, fb_prompt, raw_fb, usage_fb, fb_parsed
        )
        if fb_parsed and "question" in fb_parsed:
            current_qac  = fb_parsed
            did_fallback = True
            print("    [Step 3b] done — QAC updated")
        else:
            print("    [Step 3b] parse failed — keeping current QAC")

    # ── Step 3: Standard refine ───────────────────────────────────────────────
    elif has_issues:
        print("    [Step 3] Refining ...", end=" ", flush=True)
        ref_prompt = build_refine_prompt(current_qac, crit_parsed)
        raw_r, usage_r = call_llm(
            [{"role": "system", "content": REFINE_SYSTEM},
             {"role": "user",   "content": ref_prompt}],
            model=generation_model, api_key=api_key,
            max_tokens=cfg.get("max_tokens_refine", 4000),
            connect_timeout=cfg.get("http_connect_timeout", 10),
            read_timeout=cfg.get("http_read_timeout", 180),
            retries=cfg.get("max_retries", 3),
            retry_delay=cfg.get("retry_delay_base", 10),
        )
        ref_parsed = parse_json_response(raw_r) if raw_r else None
        if not isinstance(ref_parsed, dict):
            ref_parsed = None
        qac_log["stages"]["refine"] = _stage_entry(
            "refine", generation_model, ref_prompt, raw_r, usage_r, ref_parsed
        )
        if ref_parsed and "question" in ref_parsed:
            current_qac = ref_parsed
            print("done")
        else:
            print("parse failed — keeping current QAC")
    else:
        print("    [Step 3] No issues — skipping refine")

    # ── Step 4: First verification ────────────────────────────────────────────
    print("    [Step 4] Verification ...")
    v1_results, flagged, flag_reasons = _run_verifiers(
        current_qac, verifier_1, verifier_2, api_key, cfg, "Step 4"
    )
    qac_log["stages"]["verify_1"] = {
        "results": v1_results, "flagged": flagged, "flag_reasons": flag_reasons,
    }

    if not flagged:
        print("    -> PASSED (pipeline)")
        pipeline_status = "passed"
        qac_log["final_qac"] = current_qac
        qac_log["status"]    = "manual_review" if substring_failed else pipeline_status
        return current_qac, qac_log

    # ── Step 5: Second refine (skip if fallback already ran) ──────────────────
    if did_fallback:
        print("    [Step 5] Skipping — fallback already ran")
        qac_log["final_qac"] = current_qac
        qac_log["status"]    = "manual_review"
        return current_qac, qac_log

    print("    [Step 5] Second refine (verifier flags) ...", end=" ", flush=True)

    # Map verifier flag reasons back into the critique schema for the refiner
    v2_critique = {
        "phrasing_issue": False,    "phrasing_note": "",
        "terminology_mismatch": False, "terminology_note": "",
        "chunks_redundant": [],
        "chunks_insufficient": False, "chunks_insufficient_note": "",
        "rubric_issue": False,      "rubric_note": "",
        "mock_answer_issue": False, "mock_answer_note": "",
    }
    for reason in flag_reasons:
        r = reason.lower()
        if "rubric" in r:
            v2_critique["rubric_issue"]   = True
            v2_critique["rubric_note"]   += reason + " "
        elif "satisfies" in r or "mock_answer" in r:
            v2_critique["mock_answer_issue"]  = True
            v2_critique["mock_answer_note"]  += reason + " "
        elif "derivable" in r or "correct" in r:
            v2_critique["mock_answer_issue"]  = True
            v2_critique["mock_answer_note"]  += reason + " "
        else:
            v2_critique["phrasing_issue"]  = True
            v2_critique["phrasing_note"]  += reason + " "

    ref2_prompt = build_refine_prompt(current_qac, v2_critique)
    raw_r2, usage_r2 = call_llm(
        [{"role": "system", "content": REFINE_SYSTEM},
         {"role": "user",   "content": ref2_prompt}],
        model=generation_model, api_key=api_key,
        max_tokens=cfg.get("max_tokens_refine", 4000),
        connect_timeout=cfg.get("http_connect_timeout", 10),
        read_timeout=cfg.get("http_read_timeout", 180),
        retries=cfg.get("max_retries", 3),
        retry_delay=cfg.get("retry_delay_base", 10),
    )
    ref2_parsed = parse_json_response(raw_r2) if raw_r2 else None
    if not isinstance(ref2_parsed, dict):
        ref2_parsed = None
    qac_log["stages"]["refine_2"] = _stage_entry(
        "refine_2", generation_model, ref2_prompt, raw_r2, usage_r2, ref2_parsed
    )
    if ref2_parsed and "question" in ref2_parsed:
        current_qac = ref2_parsed
        print("done")
    else:
        print("parse failed — keeping current QAC")

    # ── Step 6: Final verification ────────────────────────────────────────────
    print("    [Step 6] Final verification ...")
    v2_results, flagged_2, flag_reasons_2 = _run_verifiers(
        current_qac, verifier_1, verifier_2, api_key, cfg, "Step 6"
    )
    qac_log["stages"]["verify_2"] = {
        "results": v2_results, "flagged": flagged_2, "flag_reasons": flag_reasons_2,
    }

    if flagged_2:
        print("    -> Still flagged after Step 6")
        for r in flag_reasons_2:
            print(f"       * {r}")
        pipeline_status = "manual_review"
    else:
        print("    -> PASSED (after refinement)")
        pipeline_status = "passed"

    # Substring failure always overrides to manual_review
    final_status = "manual_review" if substring_failed else pipeline_status
    if substring_failed and pipeline_status == "passed":
        print("    -> Overriding to manual_review (substring check failed)")

    qac_log["final_qac"] = current_qac
    qac_log["status"]    = final_status
    return current_qac, qac_log