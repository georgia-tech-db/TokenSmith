"""
Iterative evidence-curation agent with keyword pre-filtering and GREP tool.

Flow:
  1. Retrieve + rerank chunks, split into passages with section headers
  2. Extract keywords from question, score passages by keyword overlap
  3. Show only keyword-matching passages to LLM (needle pre-located)
  4. Agent curates: SELECT / GREP / DROP / RETRIEVE / DONE
     - GREP searches ALL passages (including hidden non-matches) for a term
  5. Synthesize cited answer from curated evidence

Total LLM calls: ≤ max_curate_steps + 1 (synthesize) ≤ 5
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
class Reference:
    """An evidence passage with section provenance."""
    id: int
    passage: str
    section: str
    source: str


@dataclass
class AgentResult:
    """Full output of the curation agent."""
    references: list[Reference]
    answer: str
    iterations: list[dict] = field(default_factory=list)
    total_llm_calls: int = 0


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
    """Extract meaningful terms from the question.

    Prioritizes proper nouns and technical terms (contain uppercase or digits)
    over generic words.
    """
    words = re.findall(r"[A-Za-z][A-Za-z0-9_]*", question)
    keywords: list[str] = []
    seen: set[str] = set()
    for w in words:
        low = w.lower()
        if low in _STOPWORDS or low in seen or len(w) < 3:
            continue
        seen.add(low)
        keywords.append(w)

    # If we have technical terms (mixed case / contain digits), prioritize them
    technical = [k for k in keywords if k != k.lower() or any(c.isdigit() for c in k)]
    if technical:
        return technical
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


def _grep_pool(pool: list[tuple[int, str, str, str]], term: str) -> list[int]:
    """Search ALL passages for a term, return matching IDs."""
    term_lower = term.lower()
    return [gid for gid, text, _, _ in pool if term_lower in text.lower()]


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


def _prepare_pool(
    nodes: list[NodeWithScore],
) -> list[tuple[int, str, str, str]]:
    """Convert nodes into (global_id, passage, section, source)."""
    pool: list[tuple[int, str, str, str]] = []
    seen: set[str] = set()
    gid = 1
    for node in nodes:
        source = node.metadata.get("file_name", "chunk")
        section = _extract_section(node.text)
        for passage in _split_passages(node.text):
            key = passage[:100]
            if key not in seen:
                seen.add(key)
                pool.append((gid, passage, section, source))
                gid += 1
    return pool


def _add_to_pool(
    pool: list[tuple[int, str, str, str]],
    nodes: list[NodeWithScore],
) -> None:
    """Append new unique passages from nodes into the pool."""
    existing = {p[:100] for _, p, _, _ in pool}
    max_id = max(gid for gid, _, _, _ in pool) if pool else 0
    for node in nodes:
        source = node.metadata.get("file_name", "chunk")
        section = _extract_section(node.text)
        for passage in _split_passages(node.text):
            key = passage[:100]
            if key not in existing:
                existing.add(key)
                max_id += 1
                pool.append((max_id, passage, section, source))


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
  DONE — finish selection

Rules:
- A passage must specifically discuss the topic asked about — not merely share a keyword
- Do NOT select exercise questions, index entries, practice problems, or figure captions
- Do NOT select passages about clearly different topics
- Include passages that define, explain, or illustrate the topic and its context
- Use GREP to find passages about related concepts not yet shown
- Do a thorough and complete coverage of the topic"""

_SYNTH_SYSTEM = """\
Write a concise, accurate answer using ONLY the numbered references below. \
Cite every claim with [N]. Do not add information beyond the references. \
If the references don't fully answer the question, say what is missing."""


def _curate_user(
    question: str,
    pool: list[tuple[int, str, str, str]],
    keywords: list[str],
    selected: list[int],
    grep_results: list[tuple[str, list[int]]],
) -> str:
    """Build the curation prompt showing keyword-matching passages + grep results."""
    lines = [f"Question: {question}"]
    lines.append(f"Key terms: {', '.join(keywords)}\n")

    # Score all passages
    scored: list[tuple[int, str, str, str, int, list[str]]] = []
    for gid, text, section, source in pool:
        score, matched = _score_passage(text, keywords)
        scored.append((gid, text, section, source, score, matched))

    # Keyword-matching passages (sorted by score descending)
    matching = sorted([s for s in scored if s[4] > 0], key=lambda x: -x[4])
    shown_ids: set[int] = set()

    if matching:
        lines.append(f"Passages matching key terms ({len(matching)}):\n")
        for gid, text, section, _, _, matched in matching:
            tag = " [SELECTED]" if gid in selected else ""
            lines.append(f"[{gid}] (§ {section}) [matches: {', '.join(matched)}]{tag}")
            lines.append(f"  {text}\n")
            shown_ids.add(gid)

    # Show GREP results for passages not already visible
    for term, ids in grep_results:
        new_ids = [i for i in ids if i not in shown_ids]
        if new_ids:
            lines.append(f'GREP "{term}" found {len(ids)} passage(s):')
            for gid, text, section, source in pool:
                if gid in new_ids:
                    tag = " [SELECTED]" if gid in selected else ""
                    lines.append(f"[{gid}] (§ {section}){tag}")
                    lines.append(f"  {text}\n")
                    shown_ids.add(gid)
        elif ids:
            lines.append(f'GREP "{term}": passages {ids} (already shown)')

    hidden = len(pool) - len(shown_ids)
    if hidden > 0:
        lines.append(f"({hidden} other passages available — use GREP to search)")

    if selected:
        lines.append(f"\nCurrently selected: {selected}")

    return "\n".join(lines)


def _synth_user(question: str, refs: list[Reference]) -> str:
    lines = [f"Question: {question}\n", "References:"]
    for r in refs:
        lines.append(f"[{r.id}] (§ {r.section}) {r.passage}")
    lines.append("\nAnswer:")
    return "\n".join(lines)


# ── Parsing ──────────────────────────────────────────────────────────────


def _parse_curate(
    response: str,
) -> tuple[list[int], list[int], list[str], str | None, bool]:
    """Parse curation response -> (add_ids, drop_ids, grep_terms, retrieve_query, done)."""
    add_ids: list[int] = []
    drop_ids: list[int] = []
    grep_terms: list[str] = []
    retrieve_query: str | None = None
    done = False

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
        elif "DONE" in upper:
            done = True

    return add_ids, drop_ids, grep_terms, retrieve_query, done


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
    max_curate_steps: int = 4,
) -> AgentResult:
    """
    Evidence-curation agent with keyword pre-filtering and GREP tool.

    1. Retrieve + rerank
    2. Split into passages, extract section headers
    3. Extract keywords, score passages — show only matches to LLM
    4. Agent curates (SELECT / GREP / DROP / RETRIEVE / DONE)
    5. Synthesize cited answer

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

    # ── 2. Keyword scoring ───────────────────────────────────────────────
    keywords = _extract_keywords(question)

    selected: list[int] = []
    valid_ids = {gid for gid, _, _, _ in pool}
    grep_results: list[tuple[str, list[int]]] = []

    # ── 3. Curate loop ───────────────────────────────────────────────────
    for step in range(max_curate_steps):
        prompt = _curate_user(question, pool, keywords, selected, grep_results)
        response = _chat(llm, _CURATE_SYSTEM, prompt)
        result.total_llm_calls += 1

        add_ids, drop_ids, grep_terms, retrieve_query, done = _parse_curate(response)

        for sid in add_ids:
            if sid in valid_ids and sid not in selected:
                selected.append(sid)
        for sid in drop_ids:
            if sid in selected:
                selected.remove(sid)

        result.iterations.append({
            "step": step + 1,
            "type": "curate",
            "response": response,
            "selected_after": list(selected),
            "grep_terms": grep_terms,
            "retrieve_query": retrieve_query,
        })

        # Execute tools
        tool_used = False

        for term in grep_terms:
            ids = _grep_pool(pool, term)
            grep_results.append((term, ids))
            tool_used = True

        if retrieve_query:
            new_bundle = QueryBundle(query_str=retrieve_query)
            new_nodes = retriever.retrieve(new_bundle)
            if reranker:
                new_nodes = reranker.postprocess_nodes(new_nodes, new_bundle)
            _add_to_pool(pool, new_nodes)
            valid_ids = {gid for gid, _, _, _ in pool}
            tool_used = True

        # Stop if done with no pending tools, or no actions at all
        if not tool_used:
            break

    # ── 4. Fallback if nothing selected ──────────────────────────────────
    if not selected:
        scored = [
            (gid, _score_passage(text, keywords)[0])
            for gid, text, _, _ in pool
        ]
        scored.sort(key=lambda x: -x[1])
        selected = [gid for gid, sc in scored[:5] if sc > 0]
        if not selected:
            selected = [gid for gid, _, _, _ in pool[:3]]

    # ── 5. Build references ──────────────────────────────────────────────
    refs: list[Reference] = []
    for gid, text, section, source in pool:
        if gid in selected:
            refs.append(Reference(
                id=len(refs) + 1,
                passage=text,
                section=section,
                source=source,
            ))

    # ── 6. Synthesize ────────────────────────────────────────────────────
    synth_prompt = _synth_user(question, refs)
    answer = _chat(llm, _SYNTH_SYSTEM, synth_prompt)
    result.total_llm_calls += 1

    result.references = refs
    result.answer = answer
    result.iterations.append({
        "step": len(result.iterations) + 1,
        "type": "synthesize",
        "num_references": len(refs),
        "answer": answer,
    })

    return result
