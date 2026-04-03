"""
Iterative evidence-curation agent with keyword enrichment.

Flow:
  1. Retrieve + rerank chunks, split into passages with section headers
  2. Extract keywords from question (automated, stopword-filtered)
  3. Keyword enrichment (1 LLM call) — expand terms with related concepts
  4. Agent curates (1-3 LLM calls): COUNT+SELECT / DROP / RETRIEVE / DONE
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
    chapter: str
    section: str
    subsection: str
    header_path: str
    source: str


@dataclass
class Reference:
    """An evidence passage with section provenance (after selection)."""
    id: int
    passage: str
    chapter: str
    section: str
    subsection: str
    header_path: str
    source: str


@dataclass
class CurateResponse:
    """Parsed output from one curator LLM turn."""
    add_ids: list[int]
    drop_ids: list[int]
    retrieve_query: str | None
    done: bool
    add_keywords: list[str]
    remove_keywords: list[str]


@dataclass
class ScoredPassage:
    """A passage with keyword-match score and matched terms."""
    id: int
    passage: str
    chapter: str
    section: str
    subsection: str
    header_path: str
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


# ── Passage preparation ─────────────────────────────────────────────────


_PAGE_MARKER = re.compile(r"---\s*Page\s+\d+\s*---")
_HEADER_RE = re.compile(r"^(#{1,4})\s+(.+)", re.MULTILINE)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _node_hierarchy(node: NodeWithScore) -> tuple[str, str, str, str]:
    md = node.metadata or {}
    chapter = md.get("chapter", "Unknown")
    section = md.get("section", chapter)
    subsection = md.get("subsection", section)
    header_path = md.get("header_path", f"{chapter} > {section} > {subsection}")
    return chapter, section, subsection, header_path


def _node_raw_text(node: NodeWithScore) -> str:
    md = node.metadata or {}
    return md.get("raw_text") or node.text or ""


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
        source = node.metadata.get("source") or node.metadata.get("file_name", "chunk")
        chapter, section, subsection, header_path = _node_hierarchy(node)
        raw_text = _node_raw_text(node)

        for text in _split_passages(raw_text):
            key = f"{source}|{header_path}|{text[:160]}"
            if key not in seen:
                seen.add(key)
                pool.append(
                    Passage(
                        id=gid,
                        passage=text,
                        chapter=chapter,
                        section=section,
                        subsection=subsection,
                        header_path=header_path,
                        source=source,
                    )
                )
                gid += 1
    return pool


def _add_to_pool(pool: list[Passage], nodes: list[NodeWithScore]) -> None:
    """Append new unique passages from nodes into the pool."""
    existing = {f"{p.source}|{p.header_path}|{p.passage[:160]}" for p in pool}
    max_id = max(p.id for p in pool) if pool else 0

    for node in nodes:
        source = node.metadata.get("source") or node.metadata.get("file_name", "chunk")
        chapter, section, subsection, header_path = _node_hierarchy(node)
        raw_text = _node_raw_text(node)

        for text in _split_passages(raw_text):
            key = f"{source}|{header_path}|{text[:160]}"
            if key not in existing:
                existing.add(key)
                max_id += 1
                pool.append(
                    Passage(
                        id=max_id,
                        passage=text,
                        chapter=chapter,
                        section=section,
                        subsection=subsection,
                        header_path=header_path,
                        source=source,
                    )
                )


# ── Prompts ──────────────────────────────────────────────────────────────


_CURATE_SYSTEM = """\
You are an evidence selector for a textbook Q&A system. Given a question and \
passages, select the passages that DIRECTLY help answer the question.

Available actions (one per line):
  RETRIEVE: "<new search query>" — fetch new chunks from the index
  ADD_KEYWORDS: <comma-separated terms> — add terms used to score/filter passages
  REMOVE_KEYWORDS: <comma-separated terms> — remove terms from scoring
  DROP: <comma-separated IDs> — remove passages from selection

To select passages, write two lines together:
  COUNT: <number>
  SELECT: <comma-separated passage IDs>
COUNT must equal the number of IDs in SELECT. End with DONE.

Rules:
- A passage must specifically discuss the topic asked about — not merely share a keyword
- Do NOT select exercise questions, index entries, practice problems, or figure captions
- Do NOT select passages about clearly different topics
- Prefer passages that define, explain, or illustrate the topic and its context
- Aim for 10-15 selected passages (good coverage without noise)
- Maximum number of passages is capped at 15, rest of the passages are skipped
- You can use RETRIEVE, ADD_KEYWORDS, REMOVE_KEYWORDS before SELECT.

Worked example (question about database checkpointing; "use" adds noise):
  REMOVE_KEYWORDS: use
  COUNT: 13
  SELECT: 2, 5, 8, 11, 12, 13, 4, 6, 21, 89, 10, 3, 28
  DONE"""

_ENRICH_SYSTEM = """\
You are a keyword refiner for a textbook Q&A system. The textbook is \
"Database System Concepts" (Silberschatz, Korth, Sudarshan, 7th ed.) \
with these chapters:

Ch 1: Introduction — DBMS, data abstraction, schemas
Ch 2: Relational Model — relations, keys, relational algebra
Ch 3: Intro to SQL — SELECT, joins, aggregates, GROUP BY, HAVING
Ch 4: Intermediate SQL — outer joins, views, integrity constraints, authorization
Ch 5: Advanced SQL — functions, procedures, triggers, recursive queries, OLAP (rollup, cube)
Ch 6: E-R Model — entities, relationships, cardinality, weak entities, specialization
Ch 7: Relational DB Design — functional dependencies, normalization (BCNF, 3NF), decomposition
Ch 8: Complex Data Types — JSON, XML, spatial data, textual data
Ch 9: Application Development — web apps, servlets, security, encryption
Ch 10: Big Data — distributed storage, MapReduce, Spark, streaming, graph databases
Ch 11: Data Analytics — data warehousing, OLAP, star schema, data mining
Ch 12: Physical Storage — disks, flash, RAID, storage interfaces
Ch 13: Data Storage Structures — file organization, buffer management, column stores
Ch 14: Indexing — B+-trees, hash indices, bitmap indices, spatial indexing
Ch 15: Query Processing — selection, sorting, join algorithms (nested-loop, hash, merge)
Ch 16: Query Optimization — equivalence rules, cost estimation, dynamic programming, materialized views
Ch 17: Transactions — ACID, serializability, isolation levels, schedules
Ch 18: Concurrency Control — locking (2PL), deadlocks, timestamps, MVCC, snapshot isolation
Ch 19: Recovery — WAL, log records, checkpoints, ARIES (LSN, PageLSN, RecLSN, DirtyPageTable)
Ch 20: DB Architectures — centralized, client-server, parallel, distributed, cloud
Ch 21: Parallel/Distributed Storage — partitioning, replication, distributed file systems
Ch 22: Parallel/Distributed Query Processing — parallel sort, parallel join, distributed queries
Ch 23: Parallel/Distributed Transactions — 2PC, 3PC, Paxos, Raft, distributed concurrency
Ch 24: Advanced Indexing — Bloom filters, LSM trees, R-trees, extendable/linear hashing
Ch 25: Advanced Application Dev — performance tuning, benchmarks (TPC), LDAP
Ch 26: Blockchain — hash chains, consensus, smart contracts, permissioned blockchains

Given a question and its initial key terms, refine the keyword list.

You may:
1. REMOVE terms that are vague fillers or would cause noise (e.g. "just", \
"could", "way"). Do NOT remove terms that name the core topic of the question.
2. ADD 3-8 additional technical terms from the chapters above that would \
appear in textbook passages answering this question.

Reply in EXACTLY this format (both lines required, leave empty if nothing):
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
) -> str:
    """Build the curation prompt showing keyword-matching passages."""
    lines = [f"Question: {question}"]
    lines.append(f"Key terms: {', '.join(keywords)}\n")

    # Score all passages
    scored: list[ScoredPassage] = []
    for p in pool:
        score, matched = _score_passage(p.passage, keywords)
        scored.append(ScoredPassage(
            id=p.id,
            passage=p.passage,
            chapter=p.chapter,
            section=p.section,
            subsection=p.subsection,
            header_path=p.header_path,
            source=p.source,
            score=score,
            matched_keywords=matched,
        ))

    # Keyword-matching passages (sorted by score descending)
    matching = sorted([s for s in scored if s.score > 0], key=lambda x: -x.score)

    if matching:
        lines.append(f"Passages matching key terms ({len(matching)}):\n")
        for s in matching:
            tag = " [SELECTED]" if s.id in selected else ""
            lines.append(
                f"[{s.id}] ({s.source}) "
                f"[Ch: {s.chapter} | Sec: {s.section} | Subsec: {s.subsection}] "
                f"[matches: {', '.join(s.matched_keywords)}]{tag}"
            )
            lines.append(f"  {s.passage}\n")

    non_matching = len(pool) - len(matching)
    if non_matching > 0:
        lines.append(f"({non_matching} other passages did not match key terms)")

    if selected:
        lines.append(f"\nCurrently selected: {selected}")

    return "\n".join(lines)


def _enrich_user(question: str, keywords: list[str]) -> str:
    return (
        f"Question: {question}\n"
        f"Initial key terms: {', '.join(keywords)}"
    )


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
        lines.append(
            f"[{r.id}] ({r.source}) "
            f"[Chapter: {r.chapter}] [Section: {r.section}] "
            f"[Subsection: {r.subsection}] [Path: {r.header_path}] "
            f"{r.passage}"
        )
    lines.append("\nAnswer:")
    return "\n".join(lines)


# ── Parsing ──────────────────────────────────────────────────────────────


_HARD_MAX_SELECT = 15


def _parse_curate(response: str) -> CurateResponse:
    """Parse curation response into a CurateResponse.

    Enforces the COUNT/SELECT contract:
    - If COUNT is provided, only the first `count` SELECT IDs are kept.
    - If COUNT is missing, all SELECT IDs are kept (up to _HARD_MAX_SELECT).
    - _HARD_MAX_SELECT is always enforced as an absolute ceiling.
    """
    add_ids: list[int] = []
    drop_ids: list[int] = []
    retrieve_query: str | None = None
    done = False
    add_keywords: list[str] = []
    remove_keywords: list[str] = []
    declared_count: int | None = None

    def _parse_comma_terms(s: str) -> list[str]:
        return [t.strip() for t in s.split(",") if t.strip()]

    for line in response.split("\n"):
        stripped = line.strip()
        upper = stripped.upper()

        if upper.startswith("COUNT"):
            nums = re.findall(r"\d+", stripped[5:])
            if nums:
                declared_count = int(nums[0])
        elif upper.startswith("SELECT"):
            nums = re.findall(r"\d+", stripped[6:])
            add_ids.extend(int(n) for n in nums)
        elif upper.startswith("DROP"):
            nums = re.findall(r"\d+", stripped[4:])
            drop_ids.extend(int(n) for n in nums)
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

    # Enforce COUNT contract: keep only first `declared_count` IDs
    if declared_count is not None and len(add_ids) > declared_count:
        print(f"  !! SELECT has {len(add_ids)} IDs but COUNT was {declared_count} → trimming")
        add_ids = add_ids[:declared_count]

    # Absolute ceiling
    if len(add_ids) > _HARD_MAX_SELECT:
        print(f"  !! SELECT has {len(add_ids)} IDs → hard-capping to {_HARD_MAX_SELECT}")
        add_ids = add_ids[:_HARD_MAX_SELECT]

    return CurateResponse(
        add_ids=add_ids,
        drop_ids=drop_ids,
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
    cfg=None,
) -> AgentResult:
    """
    Evidence-curation agent with keyword enrichment.

    1. Retrieve + rerank + grade, split into passages with section headers
    2. Extract keywords from question (automated)
    3. Agent curates (1-3 LLM calls) — COUNT+SELECT / DROP / RETRIEVE / DONE
    4. Hard cap: trim to ≤15 references
    5. Synthesize cited answer (1 LLM call)

    Returns AgentResult with references, answer, iteration log, call count.
    """
    from .retrieval_grader import grade_retrieved_nodes

    result = AgentResult(references=[], answer="")

    # ── 1. Retrieve ──────────────────────────────────────────────────────
    bundle = QueryBundle(query_str=question)
    nodes = retriever.retrieve(bundle)
    if reranker:
        nodes = reranker.postprocess_nodes(nodes, bundle)
    if cfg and cfg.use_retrieval_grader:
        nodes = grade_retrieved_nodes(nodes, question, cfg)

    pool = _prepare_pool(nodes)
    if not pool:
        result.answer = "No relevant information found in the documents."
        return result

    # ── 2. Keyword extraction ────────────────────────────────────────────
    keywords = _extract_keywords(question)

    # ── 3. Keyword enrichment (1 LLM call) ───────────────────────────────
    # enrich_prompt = _enrich_user(question, keywords)
    # enrich_response = _chat(llm, _ENRICH_SYSTEM, enrich_prompt)
    # result.total_llm_calls += 1

    # removed_terms, added_terms = _parse_enrich(enrich_response, keywords)
    # remove_lower = {t.lower() for t in removed_terms}
    # keywords = [k for k in keywords if k.lower() not in remove_lower] + added_terms
    # result.keywords = list(keywords)

    # result.iterations.append({
    #     "step": 1,
    #     "type": "enrich",
    #     "system_prompt": _ENRICH_SYSTEM,
    #     "user_prompt": enrich_prompt,
    #     "response": enrich_response,
    #     "initial_keywords": _extract_keywords(question),
    #     "removed_terms": removed_terms,
    #     "added_terms": added_terms,
    #     "keywords_after": list(keywords),
    # })

    selected: list[int] = []
    valid_ids = {p.id for p in pool}

    # ── 4. Curate loop (1-3 LLM calls) ──────────────────────────────────
    for step in range(max_curate_steps):
        prompt = _curate_user(question, pool, keywords, selected)
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
            "retrieve_query": cur.retrieve_query,
            "add_keywords": cur.add_keywords,
            "remove_keywords": cur.remove_keywords,
            "keywords_after": list(keywords),
        })

        # Execute tools
        if cur.retrieve_query:
            new_bundle = QueryBundle(query_str=cur.retrieve_query)
            new_nodes = retriever.retrieve(new_bundle)
            if reranker:
                new_nodes = reranker.postprocess_nodes(new_nodes, new_bundle)
            if cfg and cfg.use_retrieval_grader:
                new_nodes = grade_retrieved_nodes(new_nodes, cur.retrieve_query, cfg)
            _add_to_pool(pool, new_nodes)
            valid_ids = {p.id for p in pool}

        # Stop if agent says DONE, or if it only selected (no tool calls needed)
        has_tools = cur.retrieve_query or cur.add_keywords or cur.remove_keywords
        if cur.done or (cur.add_ids and not has_tools):
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
            refs.append(
                Reference(
                    id=len(refs) + 1,
                    passage=p.passage,
                    chapter=p.chapter,
                    section=p.section,
                    subsection=p.subsection,
                    header_path=p.header_path,
                    source=p.source,
                )
            )

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
