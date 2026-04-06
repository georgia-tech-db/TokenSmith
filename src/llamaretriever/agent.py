"""
BookRAG planner: classify → section-select → leaf-retrieve → synthesize.

LLM budget per query hop:
  1. Classify query type + extract entities  (1 call)
  2. Synthesize cited answer                 (1 call)
  ────────────────────────────────────────────────────
  Total: 2 LLM calls (well within the ≤5 cap)

Section selection and entity-graph expansion are non-LLM operations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from llama_index.core import QueryBundle
from llama_index.core.llms import ChatMessage, LLM, MessageRole
from llama_index.core.schema import NodeWithScore

from .config import LlamaIndexConfig
from .tree import DocumentTree


# ── Data structures ──────────────────────────────────────────────────────


@dataclass
class QueryClassification:
    query_type: str  # SINGLE_HOP | MULTI_HOP | GLOBAL
    entities: list[str]


@dataclass
class Reference:
    id: int
    passage: str
    section_id: str
    chapter: str
    section: str
    subsection: str
    header_path: str
    source: str


@dataclass
class BookRAGResult:
    references: list[Reference]
    answer: str
    query_type: str = ""
    selected_sections: list[str] = field(default_factory=list)
    total_llm_calls: int = 0
    iterations: list[dict] = field(default_factory=list)


# ── LLM helper ───────────────────────────────────────────────────────────


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _chat(llm: LLM, system: str, user: str) -> str:
    """Single LLM call. Strips Qwen3 <think> blocks."""
    resp = llm.chat([
        ChatMessage(role=MessageRole.SYSTEM, content=system),
        ChatMessage(role=MessageRole.USER, content=user + "\n/no_think"),
    ])
    text = resp.message.content.strip()
    text = _THINK_RE.sub("", text).strip()
    if "<think>" in text:
        text = text.split("</think>")[-1].strip()
        if text.startswith("<think>"):
            text = ""
    return text


# ── Classify prompt ──────────────────────────────────────────────────────


_CLASSIFY_SYSTEM = """\
You are a query classifier for a retrieval system over a database textbook.

Given a question, determine:

1. Query type:
   - SINGLE_HOP: answerable from one topic area / section
   - MULTI_HOP: requires connecting information across multiple topics or sections
   - GLOBAL: requires listing, counting, comparing, or summarizing across the \
document structure

2. Key entities: the main technical terms, concepts, algorithms, or data \
structures the question is about. Extract 2-8 terms.

Output EXACTLY this format (both lines required):
TYPE: <SINGLE_HOP|MULTI_HOP|GLOBAL>
ENTITIES: <comma-separated technical terms>"""


def _parse_classify(response: str) -> QueryClassification:
    query_type: str | None = None
    entities: list[str] = []

    for line in response.split("\n"):
        stripped = line.strip()
        upper = stripped.upper()
        if upper.startswith("TYPE"):
            rest = stripped.split(":", 1)[-1].strip().upper()
            if "MULTI" in rest:
                query_type = "MULTI_HOP"
            elif "GLOBAL" in rest:
                query_type = "GLOBAL"
            else:
                query_type = "SINGLE_HOP"
        elif upper.startswith("ENTIT"):
            rest = stripped.split(":", 1)[-1].strip()
            entities = [e.strip() for e in rest.split(",") if e.strip()]

    if query_type is None:
        print("Warning: could not parse query type, defaulting to SINGLE_HOP")
        query_type = "SINGLE_HOP"

    return QueryClassification(query_type=query_type, entities=entities)


# ── Section selection ────────────────────────────────────────────────────


def _retrieve_sections(
    section_retriever,
    question: str,
    top_k: int,
) -> list[str]:
    """Retrieve top-k section IDs via hybrid section retriever."""
    bundle = QueryBundle(query_str=question)
    nodes = section_retriever.retrieve(bundle)

    section_ids: list[str] = []
    for node in nodes:
        sid = node.metadata.get("section_id")
        if sid and sid not in section_ids:
            section_ids.append(sid)
        if len(section_ids) >= top_k:
            break

    return section_ids


def _expand_via_entities(
    tree: DocumentTree,
    entities: list[str],
    current_ids: list[str],
    max_total: int,
) -> list[str]:
    """Expand section selection using the entity graph.

    For each query entity, find sections where it appears.
    For multi-hop, also follow co-occurring entities one hop.
    """
    current_set = set(current_ids)
    expanded = list(current_ids)

    # Direct entity → section lookup
    for entity in entities:
        canonical = entity.lower()
        for sid in tree.sections_for_entity(canonical):
            if sid not in current_set and len(expanded) < max_total:
                current_set.add(sid)
                expanded.append(sid)

    # One-hop co-occurrence expansion
    seen_entities = {e.lower() for e in entities}
    for entity in entities:
        canonical = entity.lower()
        for coent in tree.cooccurring_entities(canonical):
            if coent in seen_entities:
                continue
            seen_entities.add(coent)
            for sid in tree.sections_for_entity(coent):
                if sid not in current_set and len(expanded) < max_total:
                    current_set.add(sid)
                    expanded.append(sid)

    return expanded


# ── Leaf retrieval within sections ───────────────────────────────────────


def _retrieve_section_leaves(
    tree: DocumentTree,
    section_ids: list[str],
    docstore,
    reranker,
    question: str,
    max_leaves: int,
) -> list[NodeWithScore]:
    """Collect leaf nodes from selected sections, rerank to top-k."""
    leaf_ids: list[str] = []
    seen: set[str] = set()
    for sid in section_ids:
        for lid in tree.subtree_leaf_ids(sid):
            if lid not in seen:
                seen.add(lid)
                leaf_ids.append(lid)

    nodes = [
        NodeWithScore(node=docstore.get_node(lid), score=1.0)
        for lid in leaf_ids
    ]

    if not nodes:
        raise ValueError(
            f"No leaf nodes found for sections: {section_ids}"
        )

    if reranker and len(nodes) > max_leaves:
        bundle = QueryBundle(query_str=question)
        nodes = reranker.postprocess_nodes(nodes, bundle)

    return nodes[:max_leaves]


# ── Reference building ───────────────────────────────────────────────────


def _build_references(
    nodes: list[NodeWithScore],
    tree: DocumentTree,
) -> list[Reference]:
    refs: list[Reference] = []
    for idx, nws in enumerate(nodes, start=1):
        md = nws.metadata or {}
        sid = md.get("section_id", "")
        section = tree.sections.get(sid)
        hp = section.header_path if section else []

        refs.append(Reference(
            id=idx,
            passage=nws.text or "",
            section_id=sid,
            chapter=hp[0] if hp else md.get("chapter", ""),
            section=hp[1] if len(hp) > 1 else md.get("section", ""),
            subsection=hp[2] if len(hp) > 2 else md.get("subsection", ""),
            header_path=" > ".join(hp) if hp else md.get("header_path", ""),
            source=md.get("source", ""),
        ))
    return refs


# ── Synthesis prompt ─────────────────────────────────────────────────────


_SYNTH_SYSTEM = """\
Write a concise, accurate answer using ONLY the numbered references below. \
Cite every claim with [N]. Do not add information beyond the references. \
If the references don't fully answer the question, say what is missing."""


def _synth_user(question: str, refs: list[Reference]) -> str:
    lines = [f"Question: {question}\n", "References:"]
    for r in refs:
        lines.append(
            f"[{r.id}] ({r.source}) [Path: {r.header_path}] {r.passage}"
        )
    lines.append("\nAnswer:")
    return "\n".join(lines)


# ── Main pipeline ────────────────────────────────────────────────────────


def run_bookrag(
    question: str,
    llm: LLM,
    section_retriever,
    leaf_index,
    tree: DocumentTree,
    reranker,
    cfg: LlamaIndexConfig,
) -> BookRAGResult:
    """
    BookRAG pipeline: classify → section-select → leaf-retrieve → synthesize.

    Total LLM calls: 2 (classify + synthesize).
    """
    from .retrieval_grader import grade_retrieved_nodes

    result = BookRAGResult(references=[], answer="")

    # ── 1. Classify query (1 LLM call) ───────────────────────────────────
    classify_response = _chat(llm, _CLASSIFY_SYSTEM, f"Question: {question}")
    result.total_llm_calls += 1
    classification = _parse_classify(classify_response)
    result.query_type = classification.query_type

    result.iterations.append({
        "step": 1,
        "type": "classify",
        "response": classify_response,
        "query_type": classification.query_type,
        "entities": classification.entities,
    })

    # ── 2. Section retrieval (no LLM) ────────────────────────────────────
    if classification.query_type == "GLOBAL":
        section_top_k = cfg.section_top_k * 2
    else:
        section_top_k = cfg.section_top_k

    section_ids = _retrieve_sections(section_retriever, question, section_top_k)

    # ── 3. Entity-graph expansion (no LLM) ──────────────────────────────
    if classification.query_type in ("MULTI_HOP", "GLOBAL"):
        max_total = cfg.section_top_k * 3
        section_ids = _expand_via_entities(
            tree, classification.entities, section_ids, max_total,
        )
    elif classification.entities:
        # Even for single-hop, add directly matching entity sections
        max_total = cfg.section_top_k + 3
        section_ids = _expand_via_entities(
            tree, classification.entities, section_ids, max_total,
        )

    result.selected_sections = list(section_ids)

    result.iterations.append({
        "step": 2,
        "type": "section_select",
        "retrieved_sections": [
            {"id": sid, "title": tree.sections[sid].title}
            for sid in section_ids if sid in tree.sections
        ],
        "query_type": classification.query_type,
        "entities_used": classification.entities,
    })

    # ── 4. Leaf retrieval within sections (no LLM) ──────────────────────
    leaf_nodes = _retrieve_section_leaves(
        tree, section_ids, leaf_index.docstore,
        reranker, question, cfg.max_leaves,
    )

    # ── 5. Retrieval grading (no LLM) ───────────────────────────────────
    if cfg.use_retrieval_grader:
        leaf_nodes = grade_retrieved_nodes(leaf_nodes, question, cfg)

    # ── 6. Build references ──────────────────────────────────────────────
    refs = _build_references(leaf_nodes, tree)

    if not refs:
        raise ValueError("No relevant passages found in selected sections")

    # ── 7. Synthesize (1 LLM call) ──────────────────────────────────────
    synth_prompt = _synth_user(question, refs)
    answer = _chat(llm, _SYNTH_SYSTEM, synth_prompt)
    result.total_llm_calls += 1

    result.references = refs
    result.answer = answer

    result.iterations.append({
        "step": 3,
        "type": "synthesize",
        "system_prompt": _SYNTH_SYSTEM,
        "user_prompt": synth_prompt,
        "num_references": len(refs),
        "response": answer,
    })

    return result
