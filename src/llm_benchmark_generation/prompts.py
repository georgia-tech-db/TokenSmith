"""
src/llm_benchmark_generation/prompts.py

All LLM prompt builders for the QAC generation pipeline.
Each function returns a (system_prompt, user_prompt) tuple.
"""

from __future__ import annotations


# ─────────────────────────────────────────────────────────────────────────────
# QAC Generation  (Step 1)
# ─────────────────────────────────────────────────────────────────────────────

QAC_SYSTEM = (
    "You are an expert question-answer dataset creator for a college-level "
    "database systems course. You generate precise, high-quality QAC "
    "(Question-Answer-Chunk) triplets from textbook pages. "
    "Your output must always be valid JSON with no prose outside it."
)


def build_qac_prompt(
    chapter:      int,
    window_start: int,
    window_end:   int,
    pages_text:   str,
) -> str:
    header = (
        f"You are generating QAC (Question-Answer-Chunk) triplets from pages "
        f"{window_start}-{window_end} of Chapter {chapter} of "
        "'Database System Concepts (7th edition)', a college-level database "
        "systems textbook."
    )

    difficulty_spec = """
DIFFICULTY LEVELS
=================

EASY (target 3-5 questions):
  Simple factual: "What is X?", "What does Y do?", "What does Z refer to?"
  Answered by 1-3 sentences that are sequential or within the same paragraph.

MEDIUM (target 2-3 questions):
  Relational / comparative: "How does X compare to Y for task Z?",
  "How does X use Y and why is it important?"
  Require 2-10 sentences, possibly spread across different parts of the pages.

HARD (target 1-2 questions):
  Questions that require SYNTHESISING multiple concepts and REASONING beyond
  direct text lookup. The answer must be INFERRED from the chunks — it is not
  a direct quote from any single sentence.
  CRITICAL: Reasoning must rely ONLY on facts stated in the gold chunks.
  Never require external knowledge not present in those chunks.

  Example of valid hard reasoning:
    Chunk A: "SQL is a declarative language."
    Chunk B: "Declarative languages are superior to imperative languages for
              database operations."
    Chunk C: "Python is an imperative programming language."
    Valid hard question: "Why might SQL be preferred over Python for database queries?"
    Valid answer: SQL is declarative, and declarative languages are superior to
                  imperative ones for database operations. Python is imperative,
                  so SQL is the better choice."""

    gold_chunk_rules = """
GOLD CHUNK RULES — VIOLATIONS WILL BE DETECTED AND REJECTED
=============================================================
1. A gold chunk is a SINGLE SENTENCE copied VERBATIM and CHARACTER-FOR-CHARACTER
   from the provided pages, including all punctuation, capitalisation, and spacing.
   Do NOT paraphrase, summarise, combine, or alter a single character.

2. BE GENEROUS: include every sentence that COULD plausibly be needed to answer
   the question, even if some may turn out to be redundant. It is far better to
   include one extra sentence than to omit one that is needed.
   A later refinement step will trim any truly redundant chunks.

3. The system that answers from these chunks has ZERO background knowledge.
   It will answer SOLELY from the gold chunks. Every fact needed to answer
   the question (or for hard questions: to reason to the answer) must be
   present in the gold chunks.

4. If you cannot find verbatim sentences that support a question, do not
   generate that question at all."""

    relationship_spec = """
CHUNK RELATIONSHIPS
===================
Declare relationships ONLY between gold chunks within the SAME QAC that
have a specific logical dependency:

  COMPOSITE: ALL chunks in the group are needed together — neither alone suffices.
             Example: A definition sentence AND the condition that qualifies it.

  SUBSTITUTE: ANY ONE chunk in the group is sufficient on its own.
              Example: Two sentences that each independently define the same term.

Most chunks will have no relationship — do NOT force one.
Set "composites": [] and "substitutes": [] if none exist."""

    quality_rules = """
QUALITY RULES
=============
1. Fewer high-quality questions is always better than more low-quality ones.
   Do NOT force questions where the content is thin.

2. Never invent content. Every claim in the mock answer and rubric must be
   directly traceable to the gold chunks.

3. Questions MUST be self-contained. A student with no book access must be
   able to fully understand the question from its text alone.

4. NEVER use phrases like "based on the text", "according to the passage",
   "based on the provided pages", "from the reading", or any language that
   implies the student has a specific document in front of them.
   Questions must read as if asked by a student studying from memory.

5. Use the EXACT terminology from the gold chunks in your questions and rubric.
   Do NOT substitute synonyms. If the chunk says "attribute", the question
   must say "attribute", not "component", "field", or "property".

6. Do not generate questions about page numbers, section numbers, or book structure.

7. Stay strictly within the provided pages. Do not use knowledge from other chapters."""

    output_format = """
OUTPUT FORMAT
=============
Respond with ONLY a valid JSON object. No markdown fences. No text outside the JSON.

{
  "qac_pairs": [
    {
      "difficulty": "easy" | "medium" | "hard",
      "question": "<self-contained question using exact chunk terminology>",
      "mock_answer": "<correct, well-written example answer>",
      "rubric": [
        "<criterion, e.g.: Must define X as Y using the term Z>",
        "<criterion, e.g.: Must state that A requires both B and C>"
      ],
      "gold_chunks": [
        "<VERBATIM sentence copied character-for-character from pages>",
        "<VERBATIM sentence copied character-for-character from pages>"
      ],
      "chunk_relationships": {
        "composites":  [ ["<sentence A>", "<sentence B>"] ],
        "substitutes": [ ["<sentence C>", "<sentence D>"] ]
      }
    }
  ]
}"""

    pages_block = (
        f"\nPAGES {window_start}-{window_end}\n"
        f"{'='*50}\n"
        f"{pages_text}"
    )

    return "\n".join([
        header, difficulty_spec, gold_chunk_rules,
        relationship_spec, quality_rules, output_format, pages_block,
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Self-critique  (Step 2)
# ─────────────────────────────────────────────────────────────────────────────

CRITIQUE_SYSTEM = (
    "You are a meticulous QA reviewer for a college-level database systems "
    "question-answer dataset. You identify specific, actionable problems in "
    "QAC (Question-Answer-Chunk) pairs. "
    "Your output must always be valid JSON with no prose outside it."
)


def build_critique_prompt(record: dict) -> str:
    chunks_block = "\n".join(
        f"  [{i+1}] {c}" for i, c in enumerate(record.get("gold_chunks", []))
    )
    rubric_block = "\n".join(
        f"  * {r}" for r in record.get("rubric", [])
    )

    return (
        "You are reviewing one QAC (Question-Answer-Chunk) pair from a college-level "
        "database systems textbook dataset.\n\n"
        f"DIFFICULTY : {record.get('difficulty', '').upper()}\n"
        f"QUESTION   : {record.get('question', '')}\n\n"
        f"MOCK ANSWER:\n{record.get('mock_answer', '')}\n\n"
        f"RUBRIC:\n{rubric_block}\n\n"
        f"GOLD CHUNKS:\n{chunks_block}\n\n"
        "TASK: Evaluate this QAC pair across six dimensions. For each flag set "
        "to true, your note must be SPECIFIC and ACTIONABLE — vague notes like "
        "'the answer could be clearer' are not acceptable.\n\n"
        "DIMENSIONS:\n\n"
        "1. phrasing_issue (true/false)\n"
        "   Is the question grammatically incorrect, awkwardly worded, ambiguous,\n"
        "   or confusing to a student reading it with no surrounding context?\n"
        "   If true: state the exact problem and provide a corrected version.\n\n"
        "2. terminology_mismatch (true/false)\n"
        "   Does the question, rubric, or mock answer use a word or phrase that\n"
        "   differs from the gold chunk terminology in a way that causes meaningful\n"
        "   confusion? Minor universally-understood synonyms are acceptable.\n"
        "   Non-synonyms that merely sound similar are NOT acceptable.\n"
        "   Example: 'Question uses \"components\" but gold chunk uses \"attributes\".\n"
        "   These are not synonyms.'\n\n"
        "3. chunks_redundant (list of verbatim chunk strings)\n"
        "   List any gold chunks that are NOT strictly necessary to answer the\n"
        "   question. Copy the chunk text verbatim. Leave as [] if all are needed.\n\n"
        "4. chunks_insufficient (true/false)\n"
        "   Are the gold chunks insufficient for a reader with ZERO background\n"
        "   knowledge to arrive at the correct answer (even with reasoning)?\n"
        "   If true: describe specifically what type of sentence is missing.\n\n"
        "5. rubric_issue (true/false)\n"
        "   Is any rubric criterion missing, factually wrong, contradicts the\n"
        "   gold chunks, or too vague to apply consistently?\n"
        "   If true: identify the specific criterion and what is wrong.\n\n"
        "6. mock_answer_issue (true/false)\n"
        "   Does the mock answer contain a factual error, contradict the gold\n"
        "   chunks, fail a rubric criterion, or use inconsistent terminology?\n"
        "   If true: state which part is wrong and what the correct version is.\n\n"
        "OUTPUT FORMAT (JSON only, no markdown fences):\n"
        "{\n"
        "  \"phrasing_issue\": false,\n"
        "  \"phrasing_note\": \"\",\n"
        "  \"terminology_mismatch\": false,\n"
        "  \"terminology_note\": \"\",\n"
        "  \"chunks_redundant\": [],\n"
        "  \"chunks_insufficient\": false,\n"
        "  \"chunks_insufficient_note\": \"\",\n"
        "  \"rubric_issue\": false,\n"
        "  \"rubric_note\": \"\",\n"
        "  \"mock_answer_issue\": false,\n"
        "  \"mock_answer_note\": \"\"\n"
        "}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Refinement  (Steps 3 & 3b)
# ─────────────────────────────────────────────────────────────────────────────

REFINE_SYSTEM = (
    "You are an expert editor for a college-level database systems QA dataset. "
    "You fix specific, identified problems in QAC pairs while preserving everything "
    "that is already correct. "
    "Your output must always be valid JSON with no prose outside it."
)


def build_refine_prompt(record: dict, critique: dict) -> str:
    chunks_block = "\n".join(
        f"  [{i+1}] {c}" for i, c in enumerate(record.get("gold_chunks", []))
    )
    rubric_block = "\n".join(
        f"  * {r}" for r in record.get("rubric", [])
    )

    issues = []
    if critique.get("phrasing_issue"):
        issues.append(f"PHRASING: {critique.get('phrasing_note', '')}")
    if critique.get("terminology_mismatch"):
        issues.append(f"TERMINOLOGY: {critique.get('terminology_note', '')}")
    if critique.get("chunks_redundant"):
        joined = "; ".join(f'"{c[:60]}..."' for c in critique["chunks_redundant"])
        issues.append(f"REMOVE REDUNDANT CHUNKS: {joined}")
    if critique.get("rubric_issue"):
        issues.append(f"RUBRIC: {critique.get('rubric_note', '')}")
    if critique.get("mock_answer_issue"):
        issues.append(f"MOCK ANSWER: {critique.get('mock_answer_note', '')}")

    issues_block = "\n".join(f"  - {iss}" for iss in issues) or "  (none identified)"

    return (
        "You are refining a QAC pair based on specific identified issues. "
        "Fix ONLY the listed issues. Do not change anything that was not flagged.\n\n"
        f"DIFFICULTY : {record.get('difficulty', '').upper()}\n\n"
        f"CURRENT QUESTION:\n{record.get('question', '')}\n\n"
        f"CURRENT MOCK ANSWER:\n{record.get('mock_answer', '')}\n\n"
        f"CURRENT RUBRIC:\n{rubric_block}\n\n"
        f"CURRENT GOLD CHUNKS:\n{chunks_block}\n\n"
        f"ISSUES TO FIX:\n{issues_block}\n\n"
        "RULES:\n"
        "1. Gold chunks must remain VERBATIM — you may only REMOVE chunks, never reword them.\n"
        "2. Use the exact terminology from the gold chunks in the question and rubric.\n"
        "3. Never use 'based on the text' or similar phrasing in the question.\n"
        "4. Return the complete corrected QAC — not just the changed parts.\n\n"
        "OUTPUT FORMAT (JSON only, no markdown fences):\n"
        "{\n"
        "  \"difficulty\": \"easy\" | \"medium\" | \"hard\",\n"
        "  \"question\": \"<corrected question>\",\n"
        "  \"mock_answer\": \"<corrected mock answer>\",\n"
        "  \"rubric\": [\"<criterion>\"],\n"
        "  \"gold_chunks\": [\"<VERBATIM chunk>\"],\n"
        "  \"chunk_relationships\": {\"composites\": [], \"substitutes\": []}\n"
        "}"
    )


def build_fallback_prompt(record: dict, critique: dict, targeted_pages: str) -> str:
    chunks_block = "\n".join(
        f"  [{i+1}] {c}" for i, c in enumerate(record.get("gold_chunks", []))
    )
    missing_note = critique.get("chunks_insufficient_note", "")

    return (
        "A QAC pair has been identified as having INSUFFICIENT gold chunks — "
        "the existing chunks do not fully support the question.\n\n"
        f"DIFFICULTY : {record.get('difficulty', '').upper()}\n"
        f"QUESTION   : {record.get('question', '')}\n\n"
        f"CURRENT GOLD CHUNKS:\n{chunks_block}\n\n"
        f"WHAT IS MISSING:\n{missing_note}\n\n"
        "You have been given a targeted excerpt of the textbook pages most likely "
        "to contain the missing information. Either:\n"
        "  (A) Find the missing sentence and add it to gold_chunks VERBATIM, OR\n"
        "  (B) If it is not in these pages, revise the question so it is fully\n"
        "      answerable from the EXISTING gold chunks alone.\n\n"
        "RULES: gold chunks must be verbatim; no 'based on the text' phrasing; "
        "use exact chunk terminology; return the complete corrected QAC.\n\n"
        f"TARGETED PAGE EXCERPT:\n{'='*50}\n{targeted_pages}\n{'='*50}\n\n"
        "OUTPUT FORMAT (JSON only, no markdown fences):\n"
        "{\n"
        "  \"difficulty\": \"easy\" | \"medium\" | \"hard\",\n"
        "  \"question\": \"<question>\",\n"
        "  \"mock_answer\": \"<mock answer>\",\n"
        "  \"rubric\": [\"<criterion>\"],\n"
        "  \"gold_chunks\": [\"<VERBATIM chunk>\"],\n"
        "  \"chunk_relationships\": {\"composites\": [], \"substitutes\": []}\n"
        "}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Verification  (Steps 4 & 6)
# ─────────────────────────────────────────────────────────────────────────────

VERIFY_SYSTEM = (
    "You are a rigorous QA evaluator for a college-level database systems "
    "question-answer dataset. You assess QAC pairs against strict criteria. "
    "Your output must always be valid JSON with no prose outside it."
)

VERIFY_CRITERIA = [
    "relevant_to_chunks",
    "answer_correct",
    "answer_derivable",
    "mock_answer_satisfies_rubric",
    "rubric_comprehensive",
]


def build_verify_prompt(record: dict) -> str:
    is_hard = record.get("difficulty") == "hard"

    hard_note = (
        "\nIMPORTANT — THIS IS A HARD QUESTION:\n"
        "The correct answer requires REASONING over the gold chunks — combining\n"
        "facts, following a logical chain, or making a simple inference. The answer\n"
        "does NOT need to be a direct quote from any single chunk. However, every\n"
        "step of the reasoning must rely SOLELY on facts stated in the gold chunks.\n"
        "Do NOT penalise an answer for requiring inference — that is expected.\n"
        "DO penalise an answer that requires external facts not in the chunks.\n"
    ) if is_hard else ""

    chunks_block = "\n".join(
        f"  [{i+1}] {c}" for i, c in enumerate(record.get("gold_chunks", []))
    )
    rubric_block = "\n".join(
        f"  * {r}" for r in record.get("rubric", [])
    )
    derivable_note = (
        "Reasoning over chunks is allowed and expected (hard question). "
        "No external facts permitted."
        if is_hard else
        "The answer should be directly and fully supported by the gold chunks."
    )

    return (
        "You are evaluating one QAC pair from a college-level database systems "
        "textbook dataset.\n"
        f"{hard_note}\n"
        f"DIFFICULTY : {record.get('difficulty', '').upper()}\n"
        f"QUESTION   : {record.get('question', '')}\n\n"
        f"MOCK ANSWER:\n{record.get('mock_answer', '')}\n\n"
        f"RUBRIC:\n{rubric_block}\n\n"
        f"GOLD CHUNKS:\n{chunks_block}\n\n"
        "TASK: Evaluate each criterion and return 'passed', 'failed', or 'uncertain'.\n"
        "For every non-passed criterion your note must be SPECIFIC and ACTIONABLE.\n\n"
        "CRITERIA:\n\n"
        "1. relevant_to_chunks\n"
        "   Is the question genuinely grounded in and answerable from the gold chunks?\n"
        "   Fail if the question asks about a topic not covered by any gold chunk.\n\n"
        "2. answer_correct\n"
        "   Is the mock answer factually correct with respect to the gold chunks?\n"
        "   Fail if it contains a factual error or contradicts a gold chunk.\n\n"
        f"3. answer_derivable\n"
        f"   {derivable_note}\n"
        "   Fail if the answer requires a fact not present in any gold chunk.\n\n"
        "4. mock_answer_satisfies_rubric\n"
        "   Does the mock answer satisfy EVERY criterion in the rubric?\n"
        "   Fail if any rubric criterion is not met. State which criterion fails.\n\n"
        "5. rubric_comprehensive\n"
        "   Does the rubric capture ALL the key points a complete correct answer\n"
        "   must address, given the question and gold chunks?\n\n"
        "OUTPUT FORMAT (JSON only, no markdown fences):\n"
        "{\n"
        "  \"relevant_to_chunks\":           \"passed\" | \"failed\" | \"uncertain\",\n"
        "  \"answer_correct\":               \"passed\" | \"failed\" | \"uncertain\",\n"
        "  \"answer_derivable\":             \"passed\" | \"failed\" | \"uncertain\",\n"
        "  \"mock_answer_satisfies_rubric\": \"passed\" | \"failed\" | \"uncertain\",\n"
        "  \"rubric_comprehensive\":         \"passed\" | \"failed\" | \"uncertain\",\n"
        "  \"notes\": {\n"
        "    \"relevant_to_chunks\":           \"<specific explanation if not passed>\",\n"
        "    \"answer_correct\":               \"\",\n"
        "    \"answer_derivable\":             \"\",\n"
        "    \"mock_answer_satisfies_rubric\": \"\",\n"
        "    \"rubric_comprehensive\":         \"\"\n"
        "  }\n"
        "}"
    )