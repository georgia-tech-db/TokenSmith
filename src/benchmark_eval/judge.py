"""
src/benchmark_eval/judge.py

LLM-as-Judge evaluations using TokenSmith's local Qwen 2.5 GGUF model
via llama_cpp (run_llama_cpp from src.generator).

Five evaluations
----------------
1. Chunk Relevance         — is each retrieved chunk relevant to the question?
   - individual mode: one call per chunk
   - group mode: one call per 3 chunks

2. Rubric Satisfaction     — which rubric criteria does the answer satisfy?
   - individual mode: one call per criterion
   - all mode: one call for all criteria together

3. Overall Answer Correctness — -1 / 0 / 1 score
   - without_reference: question + answer + rubric only
   - with_reference:    question + answer + mock_answer + rubric

4. Gold Chunk Presence     — deterministic substring check (no LLM call)
   uses the same 4-tier normalisation from benchmark_eval.metrics

5. Answer Faithfulness     — did the answer claim things not in retrieved chunks?

Prompt format
-------------
All judge prompts use Qwen 2.5's native chat template:
    <|im_start|>system ... <|im_end|>
    <|im_start|>user   ... <|im_end|>
    <|im_start|>assistant
followed immediately by { to prime JSON output.
The model is stopped at <|im_end|> and we parse whatever JSON it produced.
"""

from __future__ import annotations

import json
import re
import sys
import pathlib
import textwrap
from typing import Any, Dict, List, Optional

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.generator import run_llama_cpp


# ─────────────────────────────────────────────────────────────────────────────
# Judge prompt infrastructure
# ─────────────────────────────────────────────────────────────────────────────

JUDGE_SYSTEM = (
    "You are a precise, impartial evaluator for a question-answering system. "
    "You always respond with valid JSON and nothing else. "
    "Do not include any prose, markdown fences, or explanations outside the JSON object."
)

# Token budget for judge calls — small since we only need structured JSON
JUDGE_MAX_TOKENS = 600


def _format_judge_prompt(user_content: str) -> str:
    """
    Format a judge prompt using Qwen 2.5's chat template.
    Ends with the opening brace { to prime JSON output from the model.
    """
    return (
        f"<|im_start|>system\n{JUDGE_SYSTEM}\n<|im_end|>\n"
        f"<|im_start|>user\n{user_content}\n<|im_end|>\n"
        f"<|im_start|>assistant\n{{"
    )


def _call_judge(prompt: str, model_path: str) -> Optional[str]:
    """
    Call the local model and return raw text. Returns None on failure.
    The prompt already ends with { so the model continues the JSON object.
    """
    try:
        result = run_llama_cpp(
            prompt=prompt,
            model_path=model_path,
            max_tokens=JUDGE_MAX_TOKENS,
            temperature=0.0,
        )
        text = result["choices"][0]["text"].strip()
        # Re-attach the opening brace we primed with
        return "{" + text
    except Exception as exc:
        print(f"    [JUDGE] LLM call failed: {exc}")
        return None


def _parse_judge_json(raw: Optional[str]) -> Optional[Dict]:
    """
    Parse JSON from judge model output.
    Handles: valid JSON, ```json fences, partial output, leading/trailing noise.
    Returns None if all parsing attempts fail.
    """
    if not raw:
        return None

    # Strip markdown fences
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE).strip()

    # Try direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Try extracting the outermost { ... }
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass

    # Try completing a truncated JSON by closing all open braces/brackets
    try:
        open_b = cleaned.count("{") - cleaned.count("}")
        open_s = cleaned.count("[") - cleaned.count("]")
        completed = cleaned + ("]" * max(0, open_s)) + ("}" * max(0, open_b))
        return json.loads(completed)
    except Exception:
        pass

    print(f"    [JUDGE] JSON parse failed. Raw (first 200): {(raw or '')[:200]}")
    return None


def _safe_verdict(parsed: Optional[Dict], key: str, valid: set, default: str) -> str:
    if not parsed:
        return default
    val = str(parsed.get(key, default)).lower().strip()
    return val if val in valid else default


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation 1 — Chunk Relevance
# ─────────────────────────────────────────────────────────────────────────────

def _chunk_relevance_prompt_individual(question: str, chunk: str) -> str:
    return textwrap.dedent(f"""
        You are evaluating whether a retrieved text chunk is relevant to a question.

        QUESTION:
        {question}

        RETRIEVED CHUNK:
        {chunk}

        Is this chunk relevant to answering the question?

        Respond with ONLY this JSON object:
        {{
          "verdict": "relevant" | "not_relevant" | "uncertain",
          "reason": "<one sentence explaining your decision>"
        }}
    """).strip()


def _chunk_relevance_prompt_group(question: str, chunks: List[str]) -> str:
    chunk_block = "\n\n".join(
        f"CHUNK {i+1}:\n{c}" for i, c in enumerate(chunks)
    )
    chunk_verdicts = ", ".join(
        f'"chunk_{i+1}": {{"verdict": "relevant|not_relevant|uncertain", "reason": "..."}}'
        for i in range(len(chunks))
    )
    return textwrap.dedent(f"""
        You are evaluating whether retrieved text chunks are relevant to a question.

        QUESTION:
        {question}

        {chunk_block}

        For each chunk, decide: relevant | not_relevant | uncertain

        Respond with ONLY this JSON object:
        {{
          {chunk_verdicts}
        }}
    """).strip()


def judge_chunk_relevance(
    question:         str,
    retrieved_chunks: List[Dict],
    model_path:       str,
    mode:             str = "group",
) -> Dict:
    """
    Run chunk relevance evaluation in the specified mode only.

    Parameters
    ----------
    mode : "individual" — one LLM call per chunk
           "group"      — one LLM call per 3 chunks (default)

    Returns
    -------
    {
      "individual": [...],   # populated only when mode="individual"
      "group":      [...]    # populated only when mode="group"
    }
    """
    individual_results: List[Dict] = []
    group_results:      List[Dict] = []

    if mode == "individual":
        for item in retrieved_chunks:
            chunk_text = item.get("content", "")
            prompt     = _format_judge_prompt(
                _chunk_relevance_prompt_individual(question, chunk_text)
            )
            raw    = _call_judge(prompt, model_path)
            parsed = _parse_judge_json(raw)
            individual_results.append({
                "chunk_id":       item.get("chunk_id"),
                "rank":           item.get("rank"),
                "content_preview": chunk_text[:100],
                "verdict":  _safe_verdict(parsed, "verdict",
                                          {"relevant", "not_relevant", "uncertain"},
                                          "uncertain"),
                "reason":   (parsed or {}).get("reason", ""),
            })

    else:  # group
        for start in range(0, len(retrieved_chunks), 3):
            group  = retrieved_chunks[start: start + 3]
            texts  = [g.get("content", "") for g in group]
            prompt = _format_judge_prompt(
                _chunk_relevance_prompt_group(question, texts)
            )
            raw    = _call_judge(prompt, model_path)
            parsed = _parse_judge_json(raw)

            for j, item in enumerate(group):
                chunk_key    = f"chunk_{j+1}"
                chunk_parsed = (parsed or {}).get(chunk_key, {})
                group_results.append({
                    "chunk_id":       item.get("chunk_id"),
                    "rank":           item.get("rank"),
                    "content_preview": item.get("content", "")[:100],
                    "verdict":  _safe_verdict(chunk_parsed, "verdict",
                                              {"relevant", "not_relevant", "uncertain"},
                                              "uncertain"),
                    "reason":   chunk_parsed.get("reason", "") if isinstance(chunk_parsed, dict) else "",
                })

    return {"individual": individual_results, "group": group_results}


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation 2 — Rubric Satisfaction
# ─────────────────────────────────────────────────────────────────────────────

def _rubric_individual_prompt(question: str, answer: str, criterion: str) -> str:
    return textwrap.dedent(f"""
        You are evaluating whether a student answer satisfies a specific rubric criterion.

        QUESTION:
        {question}

        ANSWER:
        {answer}

        RUBRIC CRITERION:
        {criterion}

        Does the answer satisfy this criterion?

        Respond with ONLY this JSON object:
        {{
          "verdict": "met" | "not_met" | "partial",
          "reason": "<one sentence explaining your decision>"
        }}
    """).strip()


def _rubric_all_prompt(question: str, answer: str, criteria: List[str]) -> str:
    crit_block = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(criteria))
    crit_keys  = ", ".join(
        f'"criterion_{i+1}": {{"verdict": "met|not_met|partial", "reason": "..."}}'
        for i in range(len(criteria))
    )
    return textwrap.dedent(f"""
        You are evaluating whether a student answer satisfies rubric criteria.

        QUESTION:
        {question}

        ANSWER:
        {answer}

        RUBRIC CRITERIA:
        {crit_block}

        For each criterion, decide: met | not_met | partial

        Respond with ONLY this JSON object:
        {{
          {crit_keys}
        }}
    """).strip()


def judge_rubric_satisfaction(
    question:   str,
    ts_answer:  str,
    rubric:     List[str],
    model_path: str,
    mode:       str = "all",
) -> Dict:
    """
    Run rubric satisfaction evaluation in the specified mode only.

    Parameters
    ----------
    mode : "individual" — one LLM call per criterion
           "all"        — one LLM call for all criteria together (default)

    Returns
    -------
    {
      "individual": [...],   # populated only when mode="individual"
      "all":        [...]    # populated only when mode="all"
    }
    """
    individual_results: List[Dict] = []
    all_results:        List[Dict] = []

    if mode == "individual":
        for crit in rubric:
            prompt = _format_judge_prompt(
                _rubric_individual_prompt(question, ts_answer, crit)
            )
            raw    = _call_judge(prompt, model_path)
            parsed = _parse_judge_json(raw)
            individual_results.append({
                "criterion": crit,
                "verdict":   _safe_verdict(parsed, "verdict",
                                           {"met", "not_met", "partial"}, "not_met"),
                "reason":    (parsed or {}).get("reason", ""),
            })

    else:  # all
        if rubric:
            prompt = _format_judge_prompt(
                _rubric_all_prompt(question, ts_answer, rubric)
            )
            raw    = _call_judge(prompt, model_path)
            parsed = _parse_judge_json(raw)

            for i, crit in enumerate(rubric):
                crit_key    = f"criterion_{i+1}"
                crit_parsed = (parsed or {}).get(crit_key, {})
                all_results.append({
                    "criterion": crit,
                    "verdict":   _safe_verdict(crit_parsed, "verdict",
                                               {"met", "not_met", "partial"}, "not_met"),
                    "reason":    crit_parsed.get("reason", "") if isinstance(crit_parsed, dict) else "",
                })

    return {"individual": individual_results, "all": all_results}


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation 3 — Overall Answer Correctness
# ─────────────────────────────────────────────────────────────────────────────

def _correctness_prompt(
    question:    str,
    ts_answer:   str,
    rubric:      List[str],
    mock_answer: Optional[str] = None,
) -> str:
    rubric_block = "\n".join(f"  - {r}" for r in rubric)
    ref_section  = (
        f"\nREFERENCE ANSWER (for comparison only — other phrasings can also be correct):\n{mock_answer}\n"
        if mock_answer else ""
    )
    return textwrap.dedent(f"""
        You are scoring the quality of a student's answer to a database systems question.

        QUESTION:
        {question}

        STUDENT ANSWER:
        {ts_answer}
        {ref_section}
        RUBRIC (key points a correct answer must address):
        {rubric_block}

        Score the student answer on a scale of -1 to 1:
          1  = fully correct — addresses all key rubric points accurately
          0  = partially correct — addresses some rubric points but misses or gets others wrong
         -1  = incorrect or completely off-topic

        Respond with ONLY this JSON object:
        {{
          "score": 1 | 0 | -1,
          "explanation": "<2-3 sentences explaining the score>",
          "strengths": ["<strength 1>", "<strength 2>"],
          "weaknesses": ["<weakness 1>", "<weakness 2>"]
        }}
    """).strip()


def judge_answer_correctness(
    question:    str,
    ts_answer:   str,
    rubric:      List[str],
    mock_answer: str,
    model_path:  str,
) -> Dict:
    """
    Score answer correctness with and without the mock answer as reference.

    Returns
    -------
    {
      "without_reference": {"score": int, "explanation": str, "strengths": [], "weaknesses": []},
      "with_reference":    {"score": int, "explanation": str, "strengths": [], "weaknesses": []}
    }
    """
    def _extract(parsed: Optional[Dict]) -> Dict:
        if not parsed:
            return {"score": 0, "explanation": "", "strengths": [], "weaknesses": []}
        raw_score = parsed.get("score", 0)
        try:
            score = int(raw_score)
            score = max(-1, min(1, score))
        except (ValueError, TypeError):
            score = 0
        return {
            "score":       score,
            "explanation": parsed.get("explanation", ""),
            "strengths":   parsed.get("strengths", []),
            "weaknesses":  parsed.get("weaknesses", []),
        }

    # Without reference mock answer
    raw_no_ref  = _call_judge(
        _format_judge_prompt(_correctness_prompt(question, ts_answer, rubric, None)),
        model_path,
    )
    # With reference mock answer
    raw_with_ref = _call_judge(
        _format_judge_prompt(_correctness_prompt(question, ts_answer, rubric, mock_answer)),
        model_path,
    )

    return {
        "without_reference": _extract(_parse_judge_json(raw_no_ref)),
        "with_reference":    _extract(_parse_judge_json(raw_with_ref)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation 4 — Gold Chunk Presence (deterministic, no LLM)
# ─────────────────────────────────────────────────────────────────────────────

def check_gold_chunk_presence(
    gold_chunks:      List[str],
    retrieved_chunks: List[Dict],   # chunks_info dicts
) -> Dict:
    """
    For each gold chunk, check if any retrieved chunk contains it as a substring
    using the same 4-tier normalisation used throughout the project.

    Returns
    -------
    {
      "coverage_rate":          float,   # fraction of gold chunks found
      "covered_gold_chunks":    [str],   # gold chunks that were found
      "missing_gold_chunks":    [str],   # gold chunks that were not found
      "per_gold_chunk":         [{gold_chunk, found, found_in_rank, found_in_chunk_preview}],
      "retrieval_precision":    float    # fraction of retrieved chunks containing any gold chunk
    }
    """
    import re as _re

    def _norm(text: str) -> str:
        return _re.sub(r"\s+", " ", text).strip()

    def _clean(text: str) -> str:
        text = _re.sub(r"^\s*[\.,!?;:\-_]+\s*$", " ", text, flags=_re.MULTILINE)
        text = _re.sub(r"---\s*Page\s+\d+\s*---", " ", text)
        text = _re.sub(r"\bPage\s+\d+\b", " ", text)
        return _norm(text)

    def _chunk_contains(gold: str, chunk_text: str) -> bool:
        norm_gold    = _norm(gold)
        stripped     = norm_gold.rstrip(".,;:!?")
        norm_chunk   = _norm(chunk_text)
        clean_chunk  = _clean(chunk_text)
        return (
            norm_gold   in norm_chunk
            or norm_gold   in clean_chunk
            or stripped in norm_chunk
            or stripped in clean_chunk
        )

    retrieved_texts = [(item.get("rank"), item.get("content", "")) for item in retrieved_chunks]

    per_gold = []
    covered  = []
    missing  = []

    for gold in gold_chunks:
        found        = False
        found_rank   = None
        found_preview = None
        for rank, rtext in retrieved_texts:
            if _chunk_contains(gold, rtext):
                found         = True
                found_rank    = rank
                found_preview = rtext[:100]
                break
        per_gold.append({
            "gold_chunk":          gold,
            "found":               found,
            "found_in_rank":       found_rank,
            "found_in_chunk_preview": found_preview,
        })
        (covered if found else missing).append(gold)

    # Retrieval precision: fraction of retrieved chunks that contain at least one gold chunk
    n_retrieved = len(retrieved_texts)
    if n_retrieved:
        gold_set = set(gold_chunks)
        n_relevant_retrieved = sum(
            1 for _, rtext in retrieved_texts
            if any(_chunk_contains(g, rtext) for g in gold_set)
        )
        precision = n_relevant_retrieved / n_retrieved
    else:
        precision = 0.0

    n_gold = len(gold_chunks)
    return {
        "coverage_rate":       len(covered) / n_gold if n_gold else 0.0,
        "covered_gold_chunks": covered,
        "missing_gold_chunks": missing,
        "per_gold_chunk":      per_gold,
        "retrieval_precision": precision,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation 5 — Answer Faithfulness
# ─────────────────────────────────────────────────────────────────────────────

def _faithfulness_prompt(ts_answer: str, retrieved_chunks: List[str]) -> str:
    chunk_block = "\n\n".join(
        f"[CHUNK {i+1}]: {c}" for i, c in enumerate(retrieved_chunks)
    )
    return textwrap.dedent(f"""
        You are evaluating whether an answer is faithful to a set of retrieved text chunks.
        An answer is faithful if every factual claim it makes can be traced back to
        information in the retrieved chunks. It is unfaithful if it introduces facts,
        numbers, or claims not present in any retrieved chunk.

        RETRIEVED CHUNKS (the only information the system had access to):
        {chunk_block}

        ANSWER TO EVALUATE:
        {ts_answer}

        Evaluate faithfulness:
          faithful           — all claims are supported by the chunks
          partially_faithful — most claims are supported but some are not
          unfaithful         — significant claims are not supported by the chunks

        Respond with ONLY this JSON object:
        {{
          "verdict": "faithful" | "partially_faithful" | "unfaithful",
          "unsupported_claims": ["<claim 1 not found in chunks>", "<claim 2>"],
          "explanation": "<2-3 sentences explaining the verdict>"
        }}
    """).strip()


def judge_faithfulness(
    ts_answer:        str,
    retrieved_chunks: List[Dict],
    model_path:       str,
) -> Dict:
    """
    Evaluate whether TokenSmith's answer is faithful to its own retrieved chunks.

    Returns
    -------
    {
      "verdict":            str,   # faithful | partially_faithful | unfaithful
      "unsupported_claims": [str],
      "explanation":        str
    }
    """
    chunk_texts = [item.get("content", "") for item in retrieved_chunks]
    if not chunk_texts:
        return {
            "verdict":            "uncertain",
            "unsupported_claims": [],
            "explanation":        "No chunks were retrieved — faithfulness cannot be evaluated.",
        }

    prompt = _format_judge_prompt(_faithfulness_prompt(ts_answer, chunk_texts))
    raw    = _call_judge(prompt, model_path)
    parsed = _parse_judge_json(raw)

    return {
        "verdict":            _safe_verdict(
            parsed, "verdict",
            {"faithful", "partially_faithful", "unfaithful", "uncertain"},
            "uncertain",
        ),
        "unsupported_claims": (parsed or {}).get("unsupported_claims", []),
        "explanation":        (parsed or {}).get("explanation", ""),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Master judge runner — runs all 5 evaluations for one result
# ─────────────────────────────────────────────────────────────────────────────

def run_all_judges(result: Dict, model_path: str, chunk_judge_mode: str = "group",
                   rubric_judge_mode: str = "all") -> Dict:
    """
    Run all 5 judge evaluations for one benchmark result dict.
    Returns a flat judgements dict to be stored alongside the result.
    """
    qac              = result["qac"]
    ts_answer        = result["ts_answer"]
    retrieved_chunks = result["retrieved_chunks"]
    question         = result["question"]
    gold_chunks      = qac.get("gold_chunks", [])
    rubric           = qac.get("rubric", [])
    mock_answer      = qac.get("mock_answer", "")

    print(f"      [JUDGE 1] Chunk relevance ...", flush=True)
    chunk_rel  = judge_chunk_relevance(question, retrieved_chunks, model_path,
                                       mode=chunk_judge_mode)

    print(f"      [JUDGE 2] Rubric satisfaction ...", flush=True)
    rubric_sat = judge_rubric_satisfaction(question, ts_answer, rubric, model_path,
                                           mode=rubric_judge_mode)

    print(f"      [JUDGE 3] Answer correctness ...", flush=True)
    correctness = judge_answer_correctness(
        question, ts_answer, rubric, mock_answer, model_path
    )

    print(f"      [JUDGE 4] Gold chunk presence (deterministic) ...", flush=True)
    gold_presence = check_gold_chunk_presence(gold_chunks, retrieved_chunks)

    print(f"      [JUDGE 5] Faithfulness ...", flush=True)
    faithfulness = judge_faithfulness(ts_answer, retrieved_chunks, model_path)

    return {
        "chunk_relevance":     chunk_rel,
        "rubric_satisfaction": rubric_sat,
        "answer_correctness":  correctness,
        "gold_chunk_presence": gold_presence,
        "faithfulness":        faithfulness,
    }