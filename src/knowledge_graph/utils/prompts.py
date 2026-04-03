KEYWORD_EXTRACTION_PROMPT = """<|im_start|>system
You are a linguistic analysis expert. Analyze the provided text and identify the {top_n} most relevant and descriptive keywords or short phrases (1-3 words).
Focus on terms that carry the most information density, such as technical terms, proper nouns, and central concepts.
Return the result as a raw JSON list of strings.
Example: ["keyword1", "phrase two", "keyword3"]
Do not include any other text or explanation in your response.<|im_end|>
<|im_start|>user
Documents:
{sample_text}
<|im_end|>
<|im_start|>assistant
"""

SYNONYM_PROMPT = """Given the following groups of keywords extracted from the corpus, 
determine which keywords within each group are true synonyms.
{groups_text}
For each group:
1. Identify sets of true synonyms (same concept, interchangeable).
2. Choose the best canonical label — prefer the form used in academic/textbook literature.
3. List keywords that are NOT synonymous with any other keyword as standalone.
Respond in JSON only:
{{
    "groups": [
        {{
            "group_id": 1,
            "synonym_groups": [
                {{"canonical": "label", "members": ["kw1", "kw2"], "reason": "..."}}
            ],
            "standalone": ["kw_x"]
        }}
    ]
}}
"""

SYNONYM_SYSTEM_PROMPT = """You are a terminology expert analyzing keywords extracted from: {corpus_description}.
Identify keywords that refer to exactly the same concept and should be merged.
"""

OPENROUTER_KEYWORD_EXTRACTION_PROMPT = """You are a linguistic analysis expert. Analyze the provided text
and identify the {top_n} most relevant and descriptive keywords
or short phrases (1-3 words). Focus on terms that carry the most
information density, such as technical terms, proper nouns, and
central concepts. Return the result as a raw JSON list of strings.
Do not include any other text or explanation in your response.
"""

SUMMARY_SYSTEM_PROMPT = """\
You are an expert at summarizing academic textbook content. \
Produce dense, accurate summaries that preserve technical terminology."""

CHUNK_SUMMARY_PROMPT = """\
Summarize the following textbook excerpt in 2-4 sentences.
Capture the key concepts, definitions, and relationships. Be precise and concise.

Text:
{text}"""

SECTION_SUMMARY_PROMPT = """\
Summarize the section "{heading}" of a textbook using the content summaries below.
Write 3-5 sentences capturing the main topics, key concepts, and their relationships.

Content:
{summaries}"""

GRADE_PROMPT = """\
You are evaluating a retrieval system for a question-answering application.

Question: {query}

Retrieved passages:
{passages}

Rate each passage for how well it helps answer the question.
Return a JSON object with key "grades" containing one entry per passage (same order):
{{"grades": [{{"id": 1, "score": 0, "reason": "brief reason"}}]}}

Scoring:
0 = Not relevant — passage is unrelated to the question
1 = Partially relevant — passage touches the topic but doesn't directly answer it
2 = Highly relevant — passage directly helps answer the question"""
