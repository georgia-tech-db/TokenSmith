Act as an expert assessment designer and evaluator. Based STRICTLY and SOLELY on the uploaded source material(s), generate an evaluation dataset of [Insert Number, e.g., 5 to 10] high-quality, open-ended questions with comprehensive answers and citations.

Do not assume, extrapolate, or introduce external knowledge. Every answer and quote must be traceable directly to the provided sources.

### Guidelines for Questions:
- **Exclusively Open-Ended:** Avoid simple "Yes/No", single-word, or multiple-choice questions. Questions must prompt analysis, explanation, comparison, or operational reasoning (e.g., "Explain how...", "Compare the mechanisms of...", "Analyze the implications of...").
- **Depth & Complexity:** Focus on core mechanisms, nuances, conditions, trade-offs, and critical takeaways rather than trivial trivia.

---

### Output Requirements:
- Output **ONLY** a single, valid JSON array containing the evaluation items.
- Do not add conversational text, introductions, or markdown formatting outside the JSON code block.
- Ensure all quotes and special characters within strings are properly escaped to prevent parsing errors.

Follow this exact JSON schema:

```json
[
  {
    "id": 1,
    "category": "Factual | Synthesis | Analysis | Scenario",
    "question": "Clear, detailed open-ended question",
    "ideal_answer": "Authoritative, thorough answer explaining the concept completely based on the source text.",
    "key_evidence_quote": "Exact verbatim quote from the text that directly proves the answer.",
    "source_reference": "Document Title / Section / Page number (if identifiable)"
  }
]