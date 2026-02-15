"""
Iterative evidence-curation agent with keyword enrichment and GREP tool.

Flow:
  1. Retrieve + rerank chunks, split into passages with section headers
  2. Extract keywords from question (automated, stopword-filtered)
  3. Keyword enrichment (1 LLM call) — expand terms with related concepts
  4. Agent curates (1-3 LLM calls): SELECT / GREP / DROP / RETRIEVE / DONE
     - GREP searches ALL passages (including hidden non-matches) for a term
  5. Hard cap: trim to ≤15 references by keyword score
  6. Synthesize cited answer (1 LLM call)

Total LLM calls: 1 (enrich) + 1-3 (curate) + 1 (synthesize) = 3-5
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from nltk.tokenize import sent_tokenize

from llama_index.core import QueryBundle
from llama_index.core.llms import ChatMessage, LLM, MessageRole
from llama_index.core.schema import NodeWithScore


# ── Data structures ──────────────────────────────────────────────────────


@dataclass
class Passage:
    """A single passage in the evidence pool (before selection)."""
    id: int
    passage: str
    section: str
    source: str


@dataclass
class Reference:
    """An evidence passage with section provenance (after selection)."""
    id: int
    passage: str
    section: str
    source: str


@dataclass
class CurateResponse:
    """Parsed output from one curator LLM turn."""
    add_ids: list[int]
    drop_ids: list[int]
    grep_terms: list[str]
    retrieve_query: str | None
    done: bool
    add_keywords: list[str]
    remove_keywords: list[str]


@dataclass
class ScoredPassage:
    """A passage with keyword-match score and matched terms."""
    id: int
    passage: str
    section: str
    source: str
    score: int
    matched_keywords: list[str]


@dataclass
class AgentResult:
    """Full output of the curation agent."""
    references: list[Reference]
    answer: str
    iterations: list[dict] = field(default_factory=list)
    total_llm_calls: int = 0
    keywords: list[str] = field(default_factory=list)  # terms used for passage scoring


# ── Keyword extraction & scoring ─────────────────────────────────────────


_STOPWORDS = frozenset({
    "a", "an", "the", "in", "on", "at", "to", "for", "of", "with", "by",
    "from", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall",
    "and", "or", "but", "not", "so", "if", "then", "than",
    "this", "that", "these", "those", "it", "its",
    "what", "which", "who", "whom", "whose", "where", "when", "why", "how",
    "all", "each", "every", "both", "few", "more", "most", "other",
    "some", "such", "no", "only", "same", "also", "use", "used", "uses",
    # Common textbook words that match too broadly
    "page", "pages", "data", "table", "value", "values", "record", "records",
    "number", "set", "start", "still", "greater", "less", "smallest",
    "largest", "first", "last", "new", "old", "two", "one", "must",
    "need", "like", "called", "example", "following", "result", "case",
    "system", "operation", "operations", "structure", "algorithm",
})


def _extract_keywords(question: str) -> list[str]:
    """Extract meaningful terms from the question (stopwords and length filtered)."""
    words = re.findall(r"[A-Za-z][A-Za-z0-9_]*", question)
    keywords: list[str] = []
    seen: set[str] = set()
    for w in words:
        low = w.lower()
        if low in _STOPWORDS or low in seen or len(w) < 3:
            continue
        seen.add(low)
        keywords.append(w)
    return keywords


def _score_passage(text: str, keywords: list[str]) -> tuple[int, list[str]]:
    """Score a passage by keyword overlap. Returns (score, matched_keywords)."""
    text_lower = text.lower()
    matched: list[str] = []
    score = 0
    for kw in keywords:
        if kw.lower() in text_lower:
            matched.append(kw)
            score += 2 if kw in text else 1  # exact-case bonus
    return score, matched


def _grep_pool(pool: list[Passage], term: str) -> list[int]:
    """Search ALL passages for a term, return matching IDs."""
    term_lower = term.lower()
    return [p.id for p in pool if term_lower in p.passage.lower()]


# ── Passage preparation ─────────────────────────────────────────────────


_PAGE_MARKER = re.compile(r"---\s*Page\s+\d+\s*---")
_HEADER_RE = re.compile(r"^(#{1,4})\s+(.+)", re.MULTILINE)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _extract_section(text: str) -> str:
    """Extract the deepest markdown section header from chunk text."""
    headers = _HEADER_RE.findall(text)
    if headers:
        return max(headers, key=lambda h: len(h[0]))[1].strip()
    return "Unknown"


def _split_passages(text: str, min_len: int = 30, max_len: int = 500) -> list[str]:
    """Split chunk text into coherent multi-sentence passages."""
    text = _PAGE_MARKER.sub("", text)
    text = re.sub(r"<!-- image -->", "", text)

    raw = re.split(r"\n\s*\n", text.strip())
    passages: list[str] = []
    buf = ""

    for para in raw:
        para = para.strip()
        if not para:
            continue
        if para.startswith("#") and len(para) < 120:
            continue
        if len(para) < min_len:
            buf = (buf + " " + para).strip() if buf else para
            continue
        if buf:
            para = buf + " " + para
            buf = ""

        if len(para) > max_len:
            sents = sent_tokenize(para)
            chunk = ""
            for sent in sents:
                if chunk and len(chunk) + len(sent) + 1 > max_len:
                    if len(chunk) >= min_len:
                        passages.append(chunk.strip())
                    chunk = sent
                else:
                    chunk = (chunk + " " + sent).strip() if chunk else sent
            if chunk and len(chunk) >= min_len:
                passages.append(chunk.strip())
        else:
            passages.append(para)

    if buf and len(buf) >= min_len:
        passages.append(buf)
    return passages


def _prepare_pool(nodes: list[NodeWithScore]) -> list[Passage]:
    """Convert nodes into a list of Passage entries."""
    pool: list[Passage] = []
    seen: set[str] = set()
    gid = 1
    for node in nodes:
        source = node.metadata.get("file_name", "chunk")
        section = _extract_section(node.text)
        for text in _split_passages(node.text):
            key = text[:100]
            if key not in seen:
                seen.add(key)
                pool.append(Passage(id=gid, passage=text, section=section, source=source))
                gid += 1
    return pool


def _add_to_pool(pool: list[Passage], nodes: list[NodeWithScore]) -> None:
    """Append new unique passages from nodes into the pool."""
    existing = {p.passage[:100] for p in pool}
    max_id = max(p.id for p in pool) if pool else 0
    for node in nodes:
        source = node.metadata.get("file_name", "chunk")
        section = _extract_section(node.text)
        for text in _split_passages(node.text):
            key = text[:100]
            if key not in existing:
                existing.add(key)
                max_id += 1
                pool.append(Passage(id=max_id, passage=text, section=section, source=source))


# ── Prompts ──────────────────────────────────────────────────────────────


_CURATE_SYSTEM = """\
You are an evidence selector for a textbook Q&A system. Given a question and \
passages pre-filtered by keyword matching, select 3-10 passages that DIRECTLY \
help answer the question.

Available tools (one per line):
  SELECT: <comma-separated passage IDs>
  GREP: "<term>" — search ALL passages for a specific term (finds hidden ones)
  DROP: <comma-separated IDs> — remove from selection
  RETRIEVE: "<new search query>" — fetch new chunks from the index
  ADD_KEYWORDS: <comma-separated terms> — add terms used to score/filter passages
  REMOVE_KEYWORDS: <comma-separated terms> — remove terms from scoring
  DONE — finish selection (only say DONE when you are done)

Rules:
- A passage must specifically discuss the topic asked about — not merely share a keyword
- Do NOT select exercise questions, index entries, practice problems, or figure captions
- Do NOT select passages about clearly different topics
- Include passages that define, explain, or illustrate the topic and its context
- Use GREP to find passages about related concepts not yet shown
- Use ADD_KEYWORDS/REMOVE_KEYWORDS to refine which passages are shown by relevance
- Aim for 3-10 selected passages (good coverage without noise)
- NEVER select more than 15 passages — if you have more, DROP the weakest ones
- You can have more than one tool call in each response.

When to use tools (use at least one when it helps before SELECT/DONE):
- REMOVE_KEYWORDS: If a key term is too broad and adds noise, remove it. Example: REMOVE_KEYWORDS: use, order
- GREP: To find passages that mention a concept not in Key terms, search all passages. Example: GREP "serializability"
- ADD_KEYWORDS: If an important term is missing from Key terms, add it. Example: ADD_KEYWORDS: normalization, B-tree
- RETRIEVE: If you need different coverage, fetch more. Example: RETRIEVE "transaction commit and rollback"

Worked example (question about database checkpointing; "use" adds noise):
  REMOVE_KEYWORDS: use
  GREP "checkpoint"
  SELECT: 2, 5, 8, 11

Then on the next turn you will see updated Key terms and any GREP results; SELECT the relevant IDs and DONE when satisfied."""

_ENRICH_SYSTEM = """\
You are a keyword refiner for a textbook Q&A system. You will be given a \
question, its initial key terms, and the section headings from retrieved \
passages so you know the domain.

Do TWO things:
1. REMOVE only vague filler words that add noise (e.g. "just", "could", \
"does", "way", "thing"). Do NOT remove domain-specific terms. Keep at least \
one initial term.
2. ADD 3-8 technical terms from the SAME domain as the sections shown that \
would help find passages answering this question.

Reply in EXACTLY this format (both lines required, even if empty):
REMOVE: term1, term2
ADD: term3, term4, term5"""

_SYNTH_SYSTEM = """\
Write a concise, accurate answer using ONLY the numbered references below. \
Cite every claim with [N]. Do not add information beyond the references. \
If the references don't fully answer the question, say what is missing."""


def _curate_user(
    question: str,
    pool: list[Passage],
    keywords: list[str],
    selected: list[int],
    grep_results: list[tuple[str, list[int]]],
) -> str:
    """Build the curation prompt showing keyword-matching passages + grep results."""
    lines = [f"Question: {question}"]
    lines.append(f"Key terms: {', '.join(keywords)}\n")

    # Score all passages
    scored: list[ScoredPassage] = []
    for p in pool:
        score, matched = _score_passage(p.passage, keywords)
        scored.append(ScoredPassage(
            id=p.id,
            passage=p.passage,
            section=p.section,
            source=p.source,
            score=score,
            matched_keywords=matched,
        ))

    # Keyword-matching passages (sorted by score descending)
    matching = sorted([s for s in scored if s.score > 0], key=lambda x: -x.score)
    shown_ids: set[int] = set()

    if matching:
        lines.append(f"Passages matching key terms ({len(matching)}):\n")
        for s in matching:
            tag = " [SELECTED]" if s.id in selected else ""
            lines.append(f"[{s.id}] (§ {s.section}) [matches: {', '.join(s.matched_keywords)}]{tag}")
            lines.append(f"  {s.passage}\n")
            shown_ids.add(s.id)

    # Show GREP results for passages not already visible
    for term, ids in grep_results:
        new_ids = [i for i in ids if i not in shown_ids]
        if new_ids:
            lines.append(f'GREP "{term}" found {len(ids)} passage(s):')
            for p in pool:
                if p.id in new_ids:
                    tag = " [SELECTED]" if p.id in selected else ""
                    lines.append(f"[{p.id}] (§ {p.section}){tag}")
                    lines.append(f"  {p.passage}\n")
                    shown_ids.add(p.id)
        elif ids:
            lines.append(f'GREP "{term}": passages {ids} (already shown)')

    hidden = len(pool) - len(shown_ids)
    if hidden > 0:
        lines.append(f"({hidden} other passages available — use GREP to search)")

    if selected:
        lines.append(f"\nCurrently selected: {selected}")

    return "\n".join(lines)


def _enrich_user(question: str, keywords: list[str], sections: list[str]) -> str:
    lines = [
        f"Question: {question}",
        f"Initial key terms: {', '.join(keywords)}",
    ]
    if sections:
        lines.append(f"Sections from retrieved passages: {', '.join(sections)}")
    return "\n".join(lines)


def _parse_enrich(response: str, existing: list[str]) -> tuple[list[str], list[str]]:
    """Parse enrichment response into (terms_to_remove, terms_to_add)."""
    existing_lower = {k.lower() for k in existing}
    remove_terms: list[str] = []
    add_terms: list[str] = []

    for line in response.split("\n"):
        stripped = line.strip()
        upper = stripped.upper()
        if upper.startswith("REMOVE"):
            rest = stripped[6:].strip().lstrip(":").strip()
            for t in rest.split(","):
                t = t.strip().strip("-").strip("•").strip("*").strip()
                if t and t.lower() in existing_lower:
                    remove_terms.append(t)
        elif upper.startswith("ADD"):
            rest = stripped[3:].strip().lstrip(":").strip()
            for t in rest.split(","):
                t = t.strip().strip("-").strip("•").strip("*").strip()
                if t and len(t) > 2 and t.lower() not in existing_lower and t.lower() not in _STOPWORDS:
                    existing_lower.add(t.lower())
                    add_terms.append(t)

    if not remove_terms and not add_terms:
        raise ValueError('Invalid response from Keyword Refiner Agent')

    return remove_terms, add_terms


def _synth_user(question: str, refs: list[Reference]) -> str:
    lines = [f"Question: {question}\n", "References:"]
    for r in refs:
        lines.append(f"[{r.id}] (§ {r.section}) {r.passage}")
    lines.append("\nAnswer:")
    return "\n".join(lines)


# ── Parsing ──────────────────────────────────────────────────────────────


def _parse_curate(response: str) -> CurateResponse:
    """Parse curation response into a CurateResponse."""
    add_ids: list[int] = []
    drop_ids: list[int] = []
    grep_terms: list[str] = []
    retrieve_query: str | None = None
    done = False
    add_keywords: list[str] = []
    remove_keywords: list[str] = []

    def _parse_comma_terms(s: str) -> list[str]:
        return [t.strip() for t in s.split(",") if t.strip()]

    for line in response.split("\n"):
        stripped = line.strip()
        upper = stripped.upper()

        if upper.startswith("SELECT"):
            nums = re.findall(r"\d+", stripped[6:])
            add_ids.extend(int(n) for n in nums)
        elif upper.startswith("DROP"):
            nums = re.findall(r"\d+", stripped[4:])
            drop_ids.extend(int(n) for n in nums)
        elif upper.startswith("GREP"):
            m = re.search(r'["\']([^"\']+)["\']', stripped)
            if m:
                grep_terms.append(m.group(1))
            else:
                term = stripped[4:].strip().strip(":").strip()
                if term:
                    grep_terms.append(term)
        elif upper.startswith("RETRIEVE"):
            m = re.search(r'["\']([^"\']+)["\']', stripped)
            if m:
                retrieve_query = m.group(1)
        elif upper.startswith("ADD_KEYWORDS"):
            rest = stripped[12:].strip().strip(":").strip()
            add_keywords.extend(_parse_comma_terms(rest))
        elif upper.startswith("REMOVE_KEYWORDS"):
            rest = stripped[15:].strip().strip(":").strip()
            remove_keywords.extend(_parse_comma_terms(rest))
        elif "DONE" in upper:
            done = True

    return CurateResponse(
        add_ids=add_ids,
        drop_ids=drop_ids,
        grep_terms=grep_terms,
        retrieve_query=retrieve_query,
        done=done,
        add_keywords=add_keywords,
        remove_keywords=remove_keywords,
    )


# ── LLM helper ───────────────────────────────────────────────────────────


def _chat(llm: LLM, system: str, user: str) -> str:
    """Single LLM call. Disables Qwen3 thinking and strips any leaked <think> blocks."""
    resp = llm.chat([
        ChatMessage(role=MessageRole.SYSTEM, content=system),
        ChatMessage(role=MessageRole.USER, content=user + "\n/no_think"),
    ])
    text = resp.message.content.strip()
    # Strip closed <think>...</think> blocks
    text = _THINK_RE.sub("", text).strip()
    # Strip unclosed <think> blocks (model ran out of tokens while thinking)
    if "<think>" in text:
        text = text.split("</think>")[-1].strip()
        if text.startswith("<think>"):
            text = ""
    return text


# ── Agent core ───────────────────────────────────────────────────────────


def run_agent(
    question: str,
    retriever,
    reranker,
    llm: LLM,
    max_curate_steps: int = 3,
) -> AgentResult:
    """
    Evidence-curation agent with keyword enrichment and GREP tool.

    1. Retrieve + rerank, split into passages with section headers
    2. Extract keywords from question (automated)
    3. Keyword enrichment (1 LLM call) — expand terms
    4. Agent curates (1-3 LLM calls) — SELECT / GREP / DROP / RETRIEVE / DONE
    5. Hard cap: trim to ≤15 references
    6. Synthesize cited answer (1 LLM call)

    Returns AgentResult with references, answer, iteration log, call count.
    """
    result = AgentResult(references=[], answer="")

    # ── 1. Retrieve ──────────────────────────────────────────────────────
    bundle = QueryBundle(query_str=question)
    nodes = retriever.retrieve(bundle)
    if reranker:
        nodes = reranker.postprocess_nodes(nodes, bundle)

    pool = _prepare_pool(nodes)
    if not pool:
        result.answer = "No relevant information found in the documents."
        return result

    # ── 2. Keyword extraction ────────────────────────────────────────────
    keywords = _extract_keywords(question)

    # ── 3. Keyword enrichment (1 LLM call) ───────────────────────────────
    sections = list(dict.fromkeys(p.section for p in pool if p.section != "Unknown"))[:10]
    enrich_prompt = _enrich_user(question, keywords, sections)
    enrich_response = _chat(llm, _ENRICH_SYSTEM, enrich_prompt)
    result.total_llm_calls += 1

    removed_terms, added_terms = _parse_enrich(enrich_response, keywords)
    remove_lower = {t.lower() for t in removed_terms}
    keywords = [k for k in keywords if k.lower() not in remove_lower] + added_terms
    result.keywords = list(keywords)

    result.iterations.append({
        "step": 1,
        "type": "enrich",
        "system_prompt": _ENRICH_SYSTEM,
        "user_prompt": enrich_prompt,
        "response": enrich_response,
        "initial_keywords": _extract_keywords(question),
        "removed_terms": removed_terms,
        "added_terms": added_terms,
        "keywords_after": list(keywords),
    })

    selected: list[int] = []
    valid_ids = {p.id for p in pool}
    grep_results: list[tuple[str, list[int]]] = []

    # ── 4. Curate loop (1-3 LLM calls) ──────────────────────────────────
    for step in range(max_curate_steps):
        prompt = _curate_user(question, pool, keywords, selected, grep_results)
        response = _chat(llm, _CURATE_SYSTEM, prompt)
        result.total_llm_calls += 1

        cur = _parse_curate(response)

        for sid in cur.add_ids:
            if sid in valid_ids and sid not in selected:
                selected.append(sid)
        for sid in cur.drop_ids:
            if sid in selected:
                selected.remove(sid)

        # Apply keyword add/remove
        remove_lower = {t.lower() for t in cur.remove_keywords}
        keywords = [k for k in keywords if k.lower() not in remove_lower]
        for term in cur.add_keywords:
            term = term.strip()
            if term and term.lower() not in {k.lower() for k in keywords}:
                keywords.append(term)

        result.iterations.append({
            "step": step + 2,  # step 1 is enrichment
            "type": "curate",
            "system_prompt": _CURATE_SYSTEM,
            "user_prompt": prompt,
            "response": response,
            "selected_after": list(selected),
            "grep_terms": cur.grep_terms,
            "retrieve_query": cur.retrieve_query,
            "add_keywords": cur.add_keywords,
            "remove_keywords": cur.remove_keywords,
            "keywords_after": list(keywords),
        })

        # Execute tools
        for term in cur.grep_terms:
            ids = _grep_pool(pool, term)
            grep_results.append((term, ids))

        if cur.retrieve_query:
            new_bundle = QueryBundle(query_str=cur.retrieve_query)
            new_nodes = retriever.retrieve(new_bundle)
            if reranker:
                new_nodes = reranker.postprocess_nodes(new_nodes, new_bundle)
            _add_to_pool(pool, new_nodes)
            valid_ids = {p.id for p in pool}

        # Stop only when agent says DONE (or we hit max steps)
        if cur.done:
            break

    result.keywords = keywords  # final set after any ADD/REMOVE_KEYWORDS

    # ── 5. Fallback if nothing selected ──────────────────────────────────
    if not selected:
        scored = [
            (p.id, _score_passage(p.passage, keywords)[0])
            for p in pool
        ]
        scored.sort(key=lambda x: -x[1])
        selected = [pid for pid, sc in scored[:5] if sc > 0]
        if not selected:
            selected = [p.id for p in pool[:3]]

    # ── Hard cap: never exceed 15 references ─────────────────────────────
    if len(selected) > 15:
        print(f"  Hard cap hit: {len(selected)} selected → trimming to 15 by keyword score")
        id_to_score: dict[int, int] = {}
        for p in pool:
            if p.id in selected:
                id_to_score[p.id] = _score_passage(p.passage, keywords)[0]
        selected = sorted(selected, key=lambda sid: -id_to_score.get(sid, 0))[:15]

    # ── 6. Build references ──────────────────────────────────────────────
    refs: list[Reference] = []
    for p in pool:
        if p.id in selected:
            refs.append(Reference(
                id=len(refs) + 1,
                passage=p.passage,
                section=p.section,
                source=p.source,
            ))

    # ── 7. Synthesize (1 LLM call) ─────────────────────────────────────
    synth_prompt = _synth_user(question, refs)
    answer = _chat(llm, _SYNTH_SYSTEM, synth_prompt)
    result.total_llm_calls += 1

    result.references = refs
    result.answer = answer
    result.iterations.append({
        "step": len(result.iterations) + 1,
        "type": "synthesize",
        "system_prompt": _SYNTH_SYSTEM,
        "user_prompt": synth_prompt,
        "num_references": len(refs),
        "response": answer,
    })

    return result
