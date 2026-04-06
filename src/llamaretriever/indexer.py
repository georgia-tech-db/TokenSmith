"""
BookRAG-style hierarchical indexing pipeline.

Pipeline:
  1. Load markdown docs from data/
  2. Parse numbered section headers (## N.N, ## N.N.N, ...) into a DocumentTree
  3. Split section content into leaf chunks
  4. Build section summaries for section-level retrieval
  5. Extract entities → entity graph (section titles + cross-references + acronyms)
  6. Build dual VectorStoreIndex (sections + leaves)
  7. Persist tree + both indices

The book markdown uses ## for ALL headers — hierarchy comes from the numbering
pattern (1.1, 1.3.1, 5.1.1.1), not from header depth.  Chapter nodes are
inferred from the leading number in each section identifier.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.llms import LLM, ChatMessage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode

from .config import LlamaIndexConfig
from .tree import DocumentTree, EntityNode, SectionNode

# Matches "## Chapter N Title" (optionally prefixed with "Page xi" etc.)
_CHAPTER_TITLE_RE = re.compile(
    r"^##\s+(?:Page\s+\S+\s+)?Chapter\s+(\d+)\s+(.+)$",
)

# Matches numbered section headers: "## 19.1  Failure Classification"
_NUMBERED_RE = re.compile(r"^##\s+(\d+(?:\.\d+)+)\s+(.+)$")

# Page markers: "## Page 908", "## Page xi", "## Page ix Something"
_PAGE_RE = re.compile(r"^##\s+Page\s", re.IGNORECASE)

_PAGE_MARKER = re.compile(r"---\s*Page\s+\d+\s*---")
_IMAGE_TAG = re.compile(r"<!-- image -->")
_ACRONYM_RE = re.compile(r"\b[A-Z][A-Z0-9]{1,}(?:[-/][A-Z0-9]+)*\b")

# End-of-chapter material — stop collecting content when we hit these
_STOP_HEADERS = frozenset({
    "review terms", "practice exercises", "exercises",
    "further reading", "bibliography", "credits", "tools",
})


# ── Markdown tree parsing ─────────────────────────────────────────────────


def _parse_markdown_files(
    filepaths: list[Path],
) -> tuple[DocumentTree, dict[str, str]]:
    """Parse markdown files into a document tree using numbered section headers.

    The book uses ``## N.N.N Title`` for all hierarchy.  Chapter nodes are
    auto-created from the leading number in section identifiers.  Content
    before the first numbered section (preamble / TOC) is skipped.

    Returns (tree, section_id → raw text content).
    """
    tree = DocumentTree()
    section_texts: dict[str, str] = {}
    counter = 0

    for filepath in filepaths:
        text = filepath.read_text(encoding="utf-8")
        source = filepath.name

        # Pass 1: extract chapter titles from TOC entries
        chapter_titles: dict[str, str] = {}
        for line in text.split("\n"):
            m = _CHAPTER_TITLE_RE.match(line)
            if m:
                chapter_titles[m.group(1)] = m.group(2).strip()

        # Pass 2: build tree from numbered sections
        chapters_created: dict[str, str] = {}  # chapter_num -> section_id
        stack: list[tuple[int, str]] = []  # (depth, section_id)
        current_id: str | None = None
        buf: list[str] = []
        collecting = False

        for line in text.split("\n"):

            m_section = _NUMBERED_RE.match(line)

            if m_section:
                # ── Numbered section header ──────────────────────────
                number = m_section.group(1)        # e.g. "19.3.1"
                title = m_section.group(2).strip()  # e.g. "Log Records"
                chapter_num = number.split(".")[0]   # e.g. "19"
                section_depth = number.count(".") + 1  # 19.1→2, 19.1.1→3

                # Skip exercise questions: numbered entries that appear after
                # end-of-chapter material for an already-parsed chapter
                if not collecting and chapter_num in chapters_created:
                    continue

                # Flush previous section, start collecting for this one
                if current_id is not None:
                    section_texts[current_id] = "\n".join(buf).strip()
                buf = []
                collecting = True

                # Auto-create chapter node if first section of this chapter
                if chapter_num not in chapters_created:
                    ch_title = chapter_titles.get(
                        chapter_num, f"Chapter {chapter_num}",
                    )
                    ch_id = f"sec_{counter}"
                    counter += 1

                    ch_node = SectionNode(
                        id=ch_id,
                        title=f"Chapter {chapter_num}: {ch_title}",
                        depth=1,
                        parent_id=None,
                        header_path=[f"Chapter {chapter_num}: {ch_title}"],
                        source=source,
                    )
                    tree.add_section(ch_node)
                    chapters_created[chapter_num] = ch_id
                    stack = [(1, ch_id)]

                # Pop to find correct parent
                while stack and stack[-1][0] >= section_depth:
                    stack.pop()
                if not stack:
                    # Safety: re-anchor to the chapter
                    stack = [(1, chapters_created[chapter_num])]

                parent_id = stack[-1][1]
                section_id = f"sec_{counter}"
                counter += 1

                header_path = [
                    tree.sections[sid].title for _, sid in stack
                ] + [f"{number} {title}"]

                section = SectionNode(
                    id=section_id,
                    title=f"{number} {title}",
                    depth=section_depth,
                    parent_id=parent_id,
                    header_path=header_path,
                    source=source,
                )
                tree.add_section(section)
                stack.append((section_depth, section_id))
                current_id = section_id

            elif line.startswith("## "):
                # ── Non-numbered ## header ────────────────────────────
                remainder = line[3:].strip()

                if _PAGE_RE.match(line):
                    # Page marker — skip but don't stop collecting
                    continue

                lower = remainder.lower()
                # Strip leading "Page NNN " that sometimes prefixes headers
                clean = re.sub(r"^page\s+\w+\s*", "", lower).strip()
                if clean in _STOP_HEADERS or lower in _STOP_HEADERS:
                    if current_id is not None:
                        section_texts[current_id] = "\n".join(buf).strip()
                        current_id = None
                        buf = []
                    collecting = False
                else:
                    if collecting:
                        buf.append(line)

            else:
                if collecting:
                    buf.append(line)

        # Flush last section
        if current_id is not None:
            section_texts[current_id] = "\n".join(buf).strip()

    return tree, section_texts


# ── Leaf chunk creation ───────────────────────────────────────────────────


def _create_leaf_nodes(
    tree: DocumentTree,
    section_texts: dict[str, str],
    splitter: SentenceSplitter,
) -> list[TextNode]:
    """Chunk each section's text into leaf TextNodes linked to the tree."""
    leaves: list[TextNode] = []

    for section_id, text in section_texts.items():
        text = text.strip()
        if not text:
            continue

        section = tree.sections[section_id]
        chunks = splitter.split_text(text)

        for chunk_idx, chunk_text in enumerate(chunks):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue

            leaf_id = f"leaf_{section_id}_{chunk_idx}"
            hp = section.header_path

            node = TextNode(
                text=chunk_text,
                id_=leaf_id,
                metadata={
                    "section_id": section_id,
                    "node_type": "leaf",
                    "depth": section.depth,
                    "header_path": " > ".join(hp),
                    "source": section.source,
                    "chapter": hp[0] if hp else section.title,
                    "section": hp[1] if len(hp) > 1 else section.title,
                    "subsection": hp[2] if len(hp) > 2 else "",
                },
                excluded_embed_metadata_keys=[
                    "section_id", "node_type", "depth",
                ],
                excluded_llm_metadata_keys=[
                    "section_id", "node_type", "depth",
                ],
            )
            leaves.append(node)
            section.leaf_ids.append(leaf_id)

    return leaves


# ── Section summary creation ──────────────────────────────────────────────


def _create_section_summaries(
    tree: DocumentTree,
    section_texts: dict[str, str],
    max_chars: int,
) -> list[TextNode]:
    """Build summary TextNodes for each section (for section-level retrieval)."""
    nodes: list[TextNode] = []

    for section_id, section in tree.sections.items():
        parts = [f"Section: {section.title}"]
        parts.append(f"Path: {' > '.join(section.header_path)}")

        if section.children:
            child_titles = [tree.sections[c].title for c in section.children]
            parts.append(f"Subsections: {', '.join(child_titles)}")

        raw = section_texts.get(section_id, "").strip()
        raw = _PAGE_MARKER.sub("", raw)
        raw = _IMAGE_TAG.sub("", raw)
        raw = raw.strip()
        if raw:
            truncated = raw[:max_chars]
            if len(raw) > max_chars:
                truncated = truncated.rsplit(" ", 1)[0] + "..."
            parts.append(truncated)

        summary = "\n".join(parts)
        section.summary = summary

        nodes.append(TextNode(
            text=summary,
            id_=f"summary_{section_id}",
            metadata={
                "section_id": section_id,
                "node_type": "section_summary",
                "depth": section.depth,
                "header_path": " > ".join(section.header_path),
                "source": section.source,
                "title": section.title,
            },
        ))

    return nodes


# ── Entity graph construction ─────────────────────────────────────────────


def _build_entity_graph(
    tree: DocumentTree,
    section_texts: dict[str, str],
) -> None:
    """Build entity graph from section titles and cross-references.

    Phase 1: each section title becomes an entity.
    Phase 2: scan section text for mentions of other section titles.
    Phase 3: extract acronyms and link to containing sections.
    """
    # Phase 1 — section titles as entities
    for sid, section in tree.sections.items():
        title = section.title.strip()
        if len(title) < 3:
            continue
        canonical = title.lower()
        if canonical not in tree.entities:
            tree.entities[canonical] = EntityNode(
                name=title, canonical=canonical, section_ids=[sid],
            )
        elif sid not in tree.entities[canonical].section_ids:
            tree.entities[canonical].section_ids.append(sid)

    # Phase 2 — cross-reference scan
    for sid, text in section_texts.items():
        text_lower = text.lower()
        for canonical, entity in tree.entities.items():
            if len(canonical) < 4:
                continue
            if canonical in text_lower and sid not in entity.section_ids:
                entity.section_ids.append(sid)

    # Phase 3 — acronyms
    for sid, text in section_texts.items():
        for match in _ACRONYM_RE.finditer(text):
            acr = match.group()
            if len(acr) < 2 or len(acr) > 12:
                continue
            canonical = acr.lower()
            if canonical not in tree.entities:
                tree.entities[canonical] = EntityNode(
                    name=acr, canonical=canonical, section_ids=[sid],
                )
            elif sid not in tree.entities[canonical].section_ids:
                tree.entities[canonical].section_ids.append(sid)


# ── LLM-based entity extraction (index-time only) ────────────────────────

_ENTITY_SYSTEM = """\
You are an expert knowledge-graph builder for a database textbook.
Given sections of text, extract the key technical entities: concepts,
algorithms, data structures, protocols, and named systems.

Output a single line:
ENTITIES: entity1, entity2, entity3, ...

Rules:
- Only extract specific, meaningful technical terms (not generic words).
- Normalize casing: use the canonical form (e.g. "B+-tree" not "b+ tree").
- Merge duplicates.
- 20-40 entities per batch is ideal.
"""


def _llm_extract_entities(
    llm: LLM,
    tree: DocumentTree,
    section_texts: dict[str, str],
) -> None:
    """Use an LLM to extract entities per chapter, merging into the tree.

    Batches all sections of each chapter into one LLM call to keep costs
    reasonable (~26 calls for a full textbook).
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
            preview = text[:600]
            batch_parts.append(f"[{section.title}]\n{preview}")

        if not batch_parts:
            continue

        user_msg = (
            f"Chapter: {ch.title}\n\n"
            + "\n\n".join(batch_parts)
        )

        response = llm.chat([
            ChatMessage(role="system", content=_ENTITY_SYSTEM),
            ChatMessage(role="user", content=user_msg),
        ])
        raw = response.message.content or ""

        for line in raw.split("\n"):
            line = line.strip()
            if line.upper().startswith("ENTITIES:"):
                entities_str = line.split(":", 1)[1].strip()
                for ent in entities_str.split(","):
                    ent = ent.strip()
                    if len(ent) < 2:
                        continue
                    canonical = ent.lower()
                    if canonical not in tree.entities:
                        tree.entities[canonical] = EntityNode(
                            name=ent,
                            canonical=canonical,
                            section_ids=list(all_sids),
                        )
                    else:
                        for sid in all_sids:
                            if sid not in tree.entities[canonical].section_ids:
                                tree.entities[canonical].section_ids.append(sid)
                break

        print(f"  LLM entities for {ch.title}: extracted from response")


# ── Index build / load ────────────────────────────────────────────────────


def build_index(
    cfg: LlamaIndexConfig,
    index_llm: LLM | None = None,
) -> tuple[VectorStoreIndex, VectorStoreIndex, DocumentTree]:
    """Build BookRAG indices: leaf index + section index + document tree."""
    print("=" * 60)
    print("Building BookRAG indices ...")
    print(f"  Data dir    : {cfg.data_dir}")
    print(f"  Persist dir : {cfg.persist_dir}")
    print(f"  Embed model : {cfg.embed_model}")
    print(f"  Chunk size  : {cfg.chunk_size}  overlap: {cfg.chunk_overlap}")
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
    tree, section_texts = _parse_markdown_files(md_files)
    print(f"Parsed {len(tree.sections)} sections")

    if not tree.sections:
        raise ValueError("No sections parsed from markdown files")

    # 3. Create leaf chunks
    splitter = SentenceSplitter(
        chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap,
    )
    leaf_nodes = _create_leaf_nodes(tree, section_texts, splitter)
    print(f"Created {len(leaf_nodes)} leaf chunks")

    if not leaf_nodes:
        raise ValueError("No leaf chunks created from sections")

    # 4. Create section summaries
    summary_nodes = _create_section_summaries(
        tree, section_texts, cfg.section_summary_chars,
    )
    print(f"Created {len(summary_nodes)} section summaries")

    # 5. Entity graph — heuristic baseline
    _build_entity_graph(tree, section_texts)
    print(f"Extracted {len(tree.entities)} heuristic entities")

    # 5b. Optional LLM-enhanced entity extraction (index-time only)
    if index_llm is not None:
        print("Running LLM-based entity extraction ...")
        before = len(tree.entities)
        _llm_extract_entities(index_llm, tree, section_texts)
        print(f"LLM added {len(tree.entities) - before} new entities")

    # 6. Build indices
    t1 = time.time()
    leaf_index = VectorStoreIndex(leaf_nodes, show_progress=True)
    print(f"Leaf index built in {time.time() - t1:.1f}s")

    t2 = time.time()
    section_index = VectorStoreIndex(summary_nodes, show_progress=True)
    print(f"Section index built in {time.time() - t2:.1f}s")

    # 7. Persist
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

    print(
        f"Loaded: {len(tree.sections)} sections, "
        f"{len(tree.entities)} entities"
    )
    return leaf_index, section_index, tree


def get_or_build_index(
    cfg: LlamaIndexConfig,
    force_rebuild: bool = False,
    index_llm: LLM | None = None,
) -> tuple[VectorStoreIndex, VectorStoreIndex, DocumentTree]:
    if (
        not force_rebuild
        and Path(cfg.leaf_persist_dir).exists()
        and Path(cfg.section_persist_dir).exists()
        and Path(cfg.tree_path).exists()
    ):
        return load_index(cfg)
    return build_index(cfg, index_llm=index_llm)
