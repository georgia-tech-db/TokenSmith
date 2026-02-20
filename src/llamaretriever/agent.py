"""
Iterative evidence-curation agent with keyword enrichment.

Flow:
  1. Retrieve + rerank chunks, split into passages with section headers
  2. Extract keywords from question (automated, stopword-filtered)
  3. Keyword enrichment (1 LLM call) — commented out
  4. Curator ranks each passage (1-3 LLM calls): RANK id=score (0=irrelevant), optional RETRIEVE, DONE
  5. Take top 15 by curator rank (or fewer if fewer have score > 0)
  6. Synthesize cited answer (1 LLM call)

Total LLM calls: 1-3 (curate) + 1 (synthesize) = 2-4
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
    retrieve_query: str | None
    done: bool
    add_keywords: list[str]
    remove_keywords: list[str]
    # id -> relevance score (0 = irrelevant); x.y one decimal for sorting
    rank_scores: dict[int, float] = field(default_factory=dict)


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
You are an evidence ranker for a textbook Q&A system. Given a question and \
passages, rank each passage by relevance to the question.

Output format (use exactly these lines):
  RANK: id1=score1, id2=score2, ...
  RETRIEVE: "<new search query>"  (optional — only if you need more passages)
  DONE

Ranking rules:
- Assign each passage ID a relevance score. Use one decimal only (e.g. 0, 1.5, 2.3, 3.0). Higher = more relevant.
- Use 0 or 0.0 for irrelevant passages (off-topic, exercises, index entries, figure captions).
- Typical scores: 0 irrelevant, 1.0 tangentially related, 2.0 relevant, 3.0 highly relevant. You may use 0–5 with one decimal (x.y).
- At most 15 passages will be kept (top 15 by score). Include every passage ID in the RANK line.

Optional: RETRIEVE to fetch more passages; then you will be called again to rank the full pool.
End with DONE."""

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
    """Build the curation prompt listing all passages for ranking."""
    lines = [f"Question: {question}"]
    lines.append(f"Key terms: {', '.join(keywords)}\n")
    lines.append("Passages to rank (assign each ID a score 0–5 with one decimal only, e.g. 2.3; 0=irrelevant):\n")
    for p in pool:
        lines.append(f"[{p.id}] (§ {p.section})")
        lines.append(f"  {p.passage}\n")
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
        lines.append(f"[{r.id}] (§ {r.section}) {r.passage}")
    lines.append("\nAnswer:")
    return "\n".join(lines)


# ── Parsing ──────────────────────────────────────────────────────────────


_HARD_MAX_SELECT = 15
# id=score with score int or one decimal (x.y)
_RANK_RE = re.compile(r"(\d+)\s*=\s*(\d+(?:\.\d)?)")


def _parse_curate(response: str) -> CurateResponse:
    """Parse curation response. Expects RANK: id=score, ... (score x.y, 0=irrelevant); optional RETRIEVE, DONE."""
    retrieve_query: str | None = None
    done = False
    rank_scores: dict[int, float] = {}

    for line in response.split("\n"):
        stripped = line.strip()
        upper = stripped.upper()
        if upper.startswith("RANK"):
            rest = stripped[4:].strip().lstrip(":").strip()
            for m in _RANK_RE.finditer(rest):
                pid = int(m.group(1))
                score = float(m.group(2))
                rank_scores[pid] = round(max(0.0, min(5.0, score)), 1)
        elif upper.startswith("RETRIEVE"):
            m = re.search(r'["\']([^"\']+)["\']', stripped)
            if m:
                retrieve_query = m.group(1)
        elif "DONE" in upper:
            done = True

    return CurateResponse(
        add_ids=[],
        drop_ids=[],
        retrieve_query=retrieve_query,
        done=done,
        add_keywords=[],
        remove_keywords=[],
        rank_scores=rank_scores,
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
    Evidence-curation agent: curator ranks each passage, we take top 15 by rank.

    1. Retrieve + rerank, split into passages
    2. Extract keywords from question (automated)
    3. Keyword enrichment (commented out)
    4. Curator ranks passages (1-3 LLM calls): RANK id=score (0=irrelevant), optional RETRIEVE, DONE
    5. Select top 15 by rank (or fewer if fewer have score > 0)
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

    # ── 4. Curate loop (1-3 LLM calls): curator ranks passages ───────────
    for step in range(max_curate_steps):
        prompt = _curate_user(question, pool, keywords, selected)
        response = _chat(llm, _CURATE_SYSTEM, prompt)
        result.total_llm_calls += 1

        cur = _parse_curate(response)

        # Selection = top 15 by curator rank (score > 0); 0 = irrelevant
        if cur.rank_scores:
            ranked = [
                (pid, cur.rank_scores[pid])
                for pid in cur.rank_scores
                if pid in valid_ids and cur.rank_scores[pid] > 0
            ]
            ranked.sort(key=lambda x: -x[1])
            selected = [pid for pid, _ in ranked[:_HARD_MAX_SELECT]]

        result.iterations.append({
            "step": step + 2,
            "type": "curate",
            "system_prompt": _CURATE_SYSTEM,
            "user_prompt": prompt,
            "response": response,
            "rank_scores": cur.rank_scores,
            "selected_after": list(selected),
            "retrieve_query": cur.retrieve_query,
            "keywords_after": list(keywords),
        })

        if cur.retrieve_query:
            new_bundle = QueryBundle(query_str=cur.retrieve_query)
            new_nodes = retriever.retrieve(new_bundle)
            if reranker:
                new_nodes = reranker.postprocess_nodes(new_nodes, new_bundle)
            _add_to_pool(pool, new_nodes)
            valid_ids = {p.id for p in pool}

        if cur.done or not cur.retrieve_query:
            break

    result.keywords = keywords

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
