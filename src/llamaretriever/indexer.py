"""
BookRAG-style hierarchical indexing pipeline.

Pipeline:
  1. Load markdown docs from data/
  2. Parse numbered section headers into a DocumentTree    (heuristics)
  3. Split section content into leaf chunks                (heuristics)
  4. Build section summaries (LLM if available, else heuristic)
  5. Extract entities → entity graph (heuristic + optional LLM)
  6. Entity resolution / canonicalization (if LLM available)
  7. Optional bottom-up parent rollup summaries
  8. Build dual VectorStoreIndex (sections + leaves)
  9. Persist tree + both indices

Heuristic primitives live in ``heuristics.py``; this module adds the
LLM-powered stages and orchestrates the full build.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode

from .config import LlamaIndexConfig
from .external_llm import ExternalLLM
from .heuristics import (
    IMAGE_TAG,
    PAGE_MARKER,
    build_entity_graph,
    create_leaf_nodes,
    create_section_summaries_heuristic,
    heuristic_summary,
    make_summary_node,
    parse_markdown_files,
)
from .tree import DocumentTree, EntityNode, Relation

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════
# LLM-GENERATED SECTION SUMMARIES
# ══════════════════════════════════════════════════════════════════════════

_SUMMARY_SYSTEM = """\
You are generating retrieval-oriented summaries for sections of a technical textbook.

For each section provided, generate a concise summary optimized for search retrieval.
Each summary must include:
- The scope (what the section covers)
- Key technical concepts, algorithms, data structures mentioned
- 2-3 example questions this section can answer

Output JSON:
{
  "summaries": [
    {
      "section_id": "<id from input>",
      "summary": "<3-5 sentence retrieval-oriented summary>"
    }
  ]
}"""


def _create_section_summaries_llm(
    llm: ExternalLLM,
    tree: DocumentTree,
    section_texts: dict[str, str],
    max_chars: int,
) -> list[TextNode]:
    """Generate LLM-powered summaries, batched per chapter.

    Falls back to heuristic for any chapter where LLM generation fails.
    """
    nodes: list[TextNode] = []
    chapters = [s for s in tree.sections.values() if s.depth == 1]
    done_ids: set[str] = set()

    for ch in chapters:
        all_sids = tree.subtree_section_ids(ch.id)
        batch_parts: list[str] = []

        for sid in all_sids:
            section = tree.sections[sid]
            raw = section_texts.get(sid, "").strip()
            raw = PAGE_MARKER.sub("", raw)
            raw = IMAGE_TAG.sub("", raw).strip()
            preview = raw[:1500] if raw else "(no body text)"

            children_str = ""
            if section.children:
                child_titles = [tree.sections[c].title for c in section.children]
                children_str = f"\nSubsections: {', '.join(child_titles)}"

            batch_parts.append(
                f"[Section ID: {sid}] {section.title}\n"
                f"Path: {' > '.join(section.header_path)}"
                f"{children_str}\n"
                f"Content:\n{preview}"
            )

        if not batch_parts:
            continue

        user_msg = (
            f"Chapter: {ch.title}\n"
            f"Number of sections: {len(batch_parts)}\n\n"
            + "\n\n---\n\n".join(batch_parts)
        )

        try:
            result = llm.generate_json(_SUMMARY_SYSTEM, user_msg)
            summaries_by_id: dict[str, str] = {}
            for item in result.get("summaries", []):
                sid = item.get("section_id", "")
                text = item.get("summary", "")
                if sid and text:
                    summaries_by_id[sid] = text

            for sid in all_sids:
                section = tree.sections[sid]
                llm_summary = summaries_by_id.get(sid)

                if llm_summary:
                    parts = [
                        f"Section: {section.title}",
                        f"Path: {' > '.join(section.header_path)}",
                    ]
                    if section.children:
                        child_titles = [tree.sections[c].title for c in section.children]
                        parts.append(f"Subsections: {', '.join(child_titles)}")
                    parts.append(llm_summary)
                    summary = "\n".join(parts)
                else:
                    raw = section_texts.get(sid, "").strip()
                    summary = heuristic_summary(section, tree, raw, max_chars)

                nodes.append(make_summary_node(section, summary))
                done_ids.add(sid)

            logger.info("LLM summaries for %s: %d/%d", ch.title, len(summaries_by_id), len(all_sids))

        except Exception:
            logger.warning(
                "LLM summary generation failed for %s, falling back to heuristic",
                ch.title,
                exc_info=True,
            )
            for sid in all_sids:
                if sid not in done_ids:
                    section = tree.sections[sid]
                    raw = section_texts.get(sid, "").strip()
                    summary = heuristic_summary(section, tree, raw, max_chars)
                    nodes.append(make_summary_node(section, summary))
                    done_ids.add(sid)

    for sid, section in tree.sections.items():
        if sid not in done_ids:
            raw = section_texts.get(sid, "").strip()
            summary = heuristic_summary(section, tree, raw, max_chars)
            nodes.append(make_summary_node(section, summary))

    return nodes


def _create_section_summaries(
    tree: DocumentTree,
    section_texts: dict[str, str],
    max_chars: int,
    llm: ExternalLLM | None = None,
) -> list[TextNode]:
    """Build section summaries — LLM-generated when available, else heuristic."""
    if llm is not None:
        return _create_section_summaries_llm(llm, tree, section_texts, max_chars)
    return create_section_summaries_heuristic(tree, section_texts, max_chars)


# ══════════════════════════════════════════════════════════════════════════
# LLM-BASED STRUCTURED ENTITY + RELATION EXTRACTION
# ══════════════════════════════════════════════════════════════════════════

_ENTITY_EXTRACT_SYSTEM = """\
You are a knowledge-graph builder for a technical textbook.

For each section provided, extract:
1. **Entities**: technical concepts, algorithms, data structures, protocols, \
named systems, metrics, properties, and operations.
2. **Relations**: typed connections between entities observed in the text.

For each entity:
- name: canonical form (e.g. "B+-tree" not "b+ tree")
- type: one of [concept, algorithm, data_structure, protocol, system, \
metric, property, operation]
- aliases: alternative names, abbreviations, acronyms used in the text
- salience: how central this entity is to the section (0.0–1.0)

For each relation:
- source: canonical entity name (must match an entity you listed)
- target: canonical entity name (must match an entity you listed)
- type: one of [is_a, part_of, uses, implements, extends, enables, \
ensures, requires, contrasts_with, optimizes, produces, stores]

Output JSON:
{
  "sections": [
    {
      "section_id": "<id>",
      "entities": [
        {"name": "...", "type": "...", "aliases": ["..."], "salience": 0.9}
      ],
      "relations": [
        {"source": "...", "target": "...", "type": "..."}
      ]
    }
  ]
}"""


def _llm_extract_entities(
    llm: ExternalLLM,
    tree: DocumentTree,
    section_texts: dict[str, str],
) -> None:
    """Schema-constrained entity + relation extraction, batched per chapter.

    Merges results into the existing entity graph built by heuristics.
    """
    chapters = [s for s in tree.sections.values() if s.depth == 1]

    for ch in chapters:
        all_sids = tree.subtree_section_ids(ch.id)
        batch_parts: list[str] = []

        for sid in all_sids:
            section = tree.sections[sid]
            text = section_texts.get(sid, "").strip()
            if not text:
                continue
            preview = text[:800]
            batch_parts.append(
                f"[Section ID: {sid}] {section.title}\n{preview}"
            )

        if not batch_parts:
            continue

        user_msg = (
            f"Chapter: {ch.title}\n\n"
            + "\n\n---\n\n".join(batch_parts)
        )

        try:
            result = llm.generate_json(_ENTITY_EXTRACT_SYSTEM, user_msg)
        except Exception:
            logger.warning(
                "LLM entity extraction failed for %s", ch.title, exc_info=True,
            )
            continue

        for sec_data in result.get("sections", []):
            sid = sec_data.get("section_id", "")
            if sid not in tree.sections:
                continue

            for ent in sec_data.get("entities", []):
                name = ent.get("name", "").strip()
                if len(name) < 2:
                    continue
                canonical = name.lower()
                aliases = [a.strip() for a in ent.get("aliases", []) if a.strip()]
                etype = ent.get("type", "")
                salience = float(ent.get("salience", 0.5))

                if canonical not in tree.entities:
                    tree.entities[canonical] = EntityNode(
                        name=name,
                        canonical=canonical,
                        section_ids=[sid],
                        aliases=aliases,
                        entity_type=etype,
                        provenance={sid: salience},
                    )
                else:
                    existing = tree.entities[canonical]
                    if sid not in existing.section_ids:
                        existing.section_ids.append(sid)
                    for a in aliases:
                        if a not in existing.aliases:
                            existing.aliases.append(a)
                    if not existing.entity_type and etype:
                        existing.entity_type = etype
                    existing.provenance[sid] = max(
                        existing.provenance.get(sid, 0.0), salience,
                    )

            for rel in sec_data.get("relations", []):
                src = rel.get("source", "").strip().lower()
                tgt = rel.get("target", "").strip().lower()
                rtype = rel.get("type", "").strip()
                if not (src and tgt and rtype):
                    continue
                if src in tree.entities:
                    tree.entities[src].relations.append(
                        Relation(
                            source=src,
                            target=tgt,
                            relation_type=rtype,
                            section_id=sid,
                        )
                    )

        logger.info("LLM entities for %s: extracted", ch.title)


# ══════════════════════════════════════════════════════════════════════════
# ENTITY RESOLUTION / CANONICALIZATION
# ══════════════════════════════════════════════════════════════════════════

_ENTITY_RESOLUTION_SYSTEM = """\
You are performing entity resolution on a knowledge graph extracted from a \
technical textbook.

Given a list of entity names, identify groups that refer to the same concept \
and should be merged. Consider:
- Abbreviations (e.g. "2PL" and "Two-Phase Locking")
- Spelling variants (e.g. "B+-tree" and "B-plus tree")
- Synonyms (e.g. "deadlock" and "circular wait")
- Acronym expansions

Output JSON:
{
  "merge_groups": [
    {
      "canonical": "<preferred name>",
      "members": ["<variant1>", "<variant2>", ...]
    }
  ]
}

Rules:
- Only include groups with 2+ members that genuinely refer to the SAME concept.
- Do NOT merge related-but-distinct concepts (e.g. "lock" vs "latch").
- The "members" list must use the exact entity names from the input."""


def _resolve_entities(
    llm: ExternalLLM,
    tree: DocumentTree,
) -> None:
    """Merge duplicate entities identified by LLM-based resolution.

    Processes entities in batches to stay within context limits.
    """
    all_names = sorted(tree.entities.keys())
    if len(all_names) < 5:
        return

    batch_size = 200
    for start in range(0, len(all_names), batch_size):
        batch = all_names[start : start + batch_size]
        user_msg = "Entity names:\n" + "\n".join(f"- {n}" for n in batch)

        try:
            result = llm.generate_json(_ENTITY_RESOLUTION_SYSTEM, user_msg)
        except Exception:
            logger.warning(
                "Entity resolution failed for batch %d–%d",
                start, start + len(batch),
                exc_info=True,
            )
            continue

        for group in result.get("merge_groups", []):
            canonical_name = group.get("canonical", "").strip()
            members = [m.strip().lower() for m in group.get("members", []) if m.strip()]
            if len(members) < 2 or not canonical_name:
                continue

            canonical_key = canonical_name.lower().strip()

            if canonical_key not in tree.entities:
                if members and members[0] in tree.entities:
                    canonical_key = members[0]
                else:
                    continue

            target = tree.entities[canonical_key]

            for member_key in members:
                if member_key == canonical_key:
                    continue
                source = tree.entities.get(member_key)
                if source is None:
                    continue

                for sid in source.section_ids:
                    if sid not in target.section_ids:
                        target.section_ids.append(sid)

                if source.name not in target.aliases and source.name.lower() != target.name.lower():
                    target.aliases.append(source.name)
                for a in source.aliases:
                    if a not in target.aliases:
                        target.aliases.append(a)

                for sid, sal in source.provenance.items():
                    target.provenance[sid] = max(
                        target.provenance.get(sid, 0.0), sal,
                    )

                for r in source.relations:
                    target.relations.append(Relation(
                        source=canonical_key,
                        target=r.target,
                        relation_type=r.relation_type,
                        section_id=r.section_id,
                    ))

                for ent in tree.entities.values():
                    for r in ent.relations:
                        if r.target == member_key:
                            r.target = canonical_key

                del tree.entities[member_key]

    logger.info("Entity resolution complete: %d entities remain", len(tree.entities))


# ══════════════════════════════════════════════════════════════════════════
# OPTIONAL PARENT ROLLUP SUMMARIES
# ══════════════════════════════════════════════════════════════════════════

_ROLLUP_SYSTEM = """\
Synthesize a parent-level summary from the child section summaries below.
Capture the overall scope and key concepts covered across all children.
Optimize for retrieval — a search system uses this to decide whether to \
examine the subtree.

Output a single paragraph, 3-5 sentences."""


def _create_parent_rollup_summaries(
    llm: ExternalLLM,
    tree: DocumentTree,
) -> None:
    """Bottom-up: replace parent section summaries with LLM rollups of child summaries.

    Only processes sections that have children (chapters and mid-level sections).
    Processes deepest parents first so that parents-of-parents get updated children.
    """
    parents = [
        s for s in tree.sections.values()
        if s.children
    ]
    parents.sort(key=lambda s: -s.depth)

    for section in parents:
        child_summaries = []
        for cid in section.children:
            child = tree.sections.get(cid)
            if child and child.summary:
                child_summaries.append(f"[{child.title}]\n{child.summary}")

        if not child_summaries:
            continue

        user_msg = (
            f"Parent section: {section.title}\n"
            f"Path: {' > '.join(section.header_path)}\n\n"
            f"Child summaries:\n\n" + "\n\n---\n\n".join(child_summaries)
        )

        try:
            rollup = llm.generate(_ROLLUP_SYSTEM, user_msg)
            parts = [
                f"Section: {section.title}",
                f"Path: {' > '.join(section.header_path)}",
            ]
            child_titles = [tree.sections[c].title for c in section.children]
            parts.append(f"Subsections: {', '.join(child_titles)}")
            parts.append(rollup)
            section.summary = "\n".join(parts)
        except Exception:
            logger.warning(
                "Parent rollup failed for %s, keeping existing summary",
                section.title,
                exc_info=True,
            )


# ══════════════════════════════════════════════════════════════════════════
# INDEX BUILD / LOAD
# ══════════════════════════════════════════════════════════════════════════


def build_index(
    cfg: LlamaIndexConfig,
    external_llm: ExternalLLM | None = None,
) -> tuple[VectorStoreIndex, VectorStoreIndex, DocumentTree]:
    """Build BookRAG indices: leaf index + section index + document tree.

    When ``external_llm`` is provided, uses it for:
      - retrieval-oriented section summaries
      - structured entity/relation extraction
      - entity resolution / canonicalization
      - parent rollup summaries
    Otherwise falls back to heuristic-only indexing.
    """
    print("=" * 60)
    print("Building BookRAG indices ...")
    print(f"  Data dir      : {cfg.data_dir}")
    print(f"  Persist dir   : {cfg.persist_dir}")
    print(f"  Embed model   : {cfg.embed_model}")
    print(f"  Chunk size    : {cfg.chunk_size}  overlap: {cfg.chunk_overlap}")
    print(f"  External LLM  : {'yes' if external_llm else 'none (heuristic only)'}")
    print("=" * 60)

    t0 = time.time()

    # 1. Find markdown files
    md_files = sorted(Path(cfg.data_dir).glob("*.md"))
    if not md_files:
        raise FileNotFoundError(
            f"No markdown files found in {cfg.data_dir}/. "
            "Run extraction first or place .md files there."
        )
    print(f"Found {len(md_files)} markdown file(s): {[f.name for f in md_files]}")

    # 2. Parse into section tree
    tree, section_texts = parse_markdown_files(md_files)
    print(f"Parsed {len(tree.sections)} sections")
    if not tree.sections:
        raise ValueError("No sections parsed from markdown files")

    # 3. Create leaf chunks
    splitter = SentenceSplitter(
        chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap,
    )
    leaf_nodes = create_leaf_nodes(tree, section_texts, splitter)
    print(f"Created {len(leaf_nodes)} leaf chunks")
    if not leaf_nodes:
        raise ValueError("No leaf chunks created from sections")

    # 4. Section summaries (LLM or heuristic)
    t_sum = time.time()
    summary_nodes = _create_section_summaries(
        tree, section_texts, cfg.section_summary_chars, llm=external_llm,
    )
    print(f"Created {len(summary_nodes)} section summaries in {time.time() - t_sum:.1f}s")

    # 5. Entity graph — heuristic baseline
    build_entity_graph(tree, section_texts)
    print(f"Extracted {len(tree.entities)} heuristic entities")

    # 5b. LLM structured entity + relation extraction
    if external_llm is not None:
        print("Running LLM-based entity/relation extraction ...")
        t_ent = time.time()
        before = len(tree.entities)
        _llm_extract_entities(external_llm, tree, section_texts)
        n_relations = sum(len(e.relations) for e in tree.entities.values())
        print(
            f"LLM added {len(tree.entities) - before} new entities, "
            f"{n_relations} relations in {time.time() - t_ent:.1f}s"
        )

    # 5c. Entity resolution / canonicalization
    if external_llm is not None:
        print("Running entity resolution ...")
        t_er = time.time()
        before = len(tree.entities)
        _resolve_entities(external_llm, tree)
        print(
            f"Entity resolution merged {before - len(tree.entities)} entities "
            f"in {time.time() - t_er:.1f}s"
        )

    # 6. Optional parent rollup summaries
    if external_llm is not None:
        print("Generating parent rollup summaries ...")
        t_ru = time.time()
        _create_parent_rollup_summaries(external_llm, tree)
        summary_nodes = [
            make_summary_node(section, section.summary)
            for section in tree.sections.values()
        ]
        print(f"Parent rollups done in {time.time() - t_ru:.1f}s")

    # 7. Build vector indices
    t1 = time.time()
    leaf_index = VectorStoreIndex(leaf_nodes, show_progress=True)
    print(f"Leaf index built in {time.time() - t1:.1f}s")

    t2 = time.time()
    section_index = VectorStoreIndex(summary_nodes, show_progress=True)
    print(f"Section index built in {time.time() - t2:.1f}s")

    # 8. Persist
    Path(cfg.leaf_persist_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.section_persist_dir).mkdir(parents=True, exist_ok=True)

    leaf_index.storage_context.persist(persist_dir=cfg.leaf_persist_dir)
    section_index.storage_context.persist(persist_dir=cfg.section_persist_dir)
    tree.save(cfg.tree_path)

    print(f"Persisted to {cfg.persist_dir}")
    print(f"Total indexing time: {time.time() - t0:.1f}s")

    return leaf_index, section_index, tree


def load_index(
    cfg: LlamaIndexConfig,
) -> tuple[VectorStoreIndex, VectorStoreIndex, DocumentTree]:
    """Load persisted BookRAG indices."""
    for path, label in [
        (cfg.leaf_persist_dir, "leaf index"),
        (cfg.section_persist_dir, "section index"),
        (cfg.tree_path, "document tree"),
    ]:
        if not Path(path).exists():
            raise FileNotFoundError(
                f"No persisted {label} at {path}. Run indexing first."
            )

    print(f"Loading indices from {cfg.persist_dir} ...")

    leaf_ctx = StorageContext.from_defaults(persist_dir=cfg.leaf_persist_dir)
    leaf_index = load_index_from_storage(leaf_ctx)

    section_ctx = StorageContext.from_defaults(persist_dir=cfg.section_persist_dir)
    section_index = load_index_from_storage(section_ctx)

    tree = DocumentTree.load(cfg.tree_path)

    n_relations = sum(len(e.relations) for e in tree.entities.values())
    print(
        f"Loaded: {len(tree.sections)} sections, "
        f"{len(tree.entities)} entities, "
        f"{n_relations} relations"
    )
    return leaf_index, section_index, tree


def get_or_build_index(
    cfg: LlamaIndexConfig,
    force_rebuild: bool = False,
    external_llm: ExternalLLM | None = None,
) -> tuple[VectorStoreIndex, VectorStoreIndex, DocumentTree]:
    if (
        not force_rebuild
        and Path(cfg.leaf_persist_dir).exists()
        and Path(cfg.section_persist_dir).exists()
        and Path(cfg.tree_path).exists()
    ):
        return load_index(cfg)
    return build_index(cfg, external_llm=external_llm)
