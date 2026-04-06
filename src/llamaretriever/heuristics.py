"""
Heuristic (non-LLM) indexing primitives for the BookRAG pipeline.

Contains:
  - Markdown tree parsing (regex-based section header detection)
  - Leaf chunk creation
  - Heuristic section summaries (truncated text)
  - Heuristic entity graph (title matching, cross-references, acronyms)
  - Shared utilities used by both heuristic and LLM paths
"""

from __future__ import annotations

import re
from pathlib import Path

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode

from .tree import DocumentTree, EntityNode, SectionNode

# ── Shared regex constants (also used by LLM paths in indexer.py) ─────────

PAGE_MARKER = re.compile(r"---\s*Page\s+\d+\s*---")
IMAGE_TAG = re.compile(r"<!-- image -->")
ACRONYM_RE = re.compile(r"\b[A-Z][A-Z0-9]{1,}(?:[-/][A-Z0-9]+)*\b")

# ── Internal regex constants ──────────────────────────────────────────────

_CHAPTER_TITLE_RE = re.compile(
    r"^##\s+(?:Page\s+\S+\s+)?Chapter\s+(\d+)\s+(.+)$",
)
_NUMBERED_RE = re.compile(r"^##\s+(\d+(?:\.\d+)+)\s+(.+)$")
_PAGE_RE = re.compile(r"^##\s+Page\s", re.IGNORECASE)

_STOP_HEADERS = frozenset({
    "review terms", "practice exercises", "exercises",
    "further reading", "bibliography", "credits", "tools",
})


# ══════════════════════════════════════════════════════════════════════════
# MARKDOWN TREE PARSING
# ══════════════════════════════════════════════════════════════════════════


def parse_markdown_files(
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

        chapter_titles: dict[str, str] = {}
        for line in text.split("\n"):
            m = _CHAPTER_TITLE_RE.match(line)
            if m:
                chapter_titles[m.group(1)] = m.group(2).strip()

        chapters_created: dict[str, str] = {}
        stack: list[tuple[int, str]] = []
        current_id: str | None = None
        buf: list[str] = []
        collecting = False

        for line in text.split("\n"):
            m_section = _NUMBERED_RE.match(line)

            if m_section:
                number = m_section.group(1)
                title = m_section.group(2).strip()
                chapter_num = number.split(".")[0]
                section_depth = number.count(".") + 1

                if not collecting and chapter_num in chapters_created:
                    continue

                if current_id is not None:
                    section_texts[current_id] = "\n".join(buf).strip()
                buf = []
                collecting = True

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

                while stack and stack[-1][0] >= section_depth:
                    stack.pop()
                if not stack:
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
                remainder = line[3:].strip()

                if _PAGE_RE.match(line):
                    continue

                lower = remainder.lower()
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

        if current_id is not None:
            section_texts[current_id] = "\n".join(buf).strip()

    return tree, section_texts


# ══════════════════════════════════════════════════════════════════════════
# LEAF CHUNK CREATION
# ══════════════════════════════════════════════════════════════════════════


def create_leaf_nodes(
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


# ══════════════════════════════════════════════════════════════════════════
# SECTION SUMMARY UTILITIES + HEURISTIC SUMMARIES
# ══════════════════════════════════════════════════════════════════════════


def make_summary_node(section: SectionNode, summary: str) -> TextNode:
    """Create a TextNode from a section summary string.

    Also sets ``section.summary`` as a side-effect so the tree stays in sync.
    """
    section.summary = summary
    return TextNode(
        text=summary,
        id_=f"summary_{section.id}",
        metadata={
            "section_id": section.id,
            "node_type": "section_summary",
            "depth": section.depth,
            "header_path": " > ".join(section.header_path),
            "source": section.source,
            "title": section.title,
        },
    )


def heuristic_summary(
    section: SectionNode,
    tree: DocumentTree,
    raw: str,
    max_chars: int,
) -> str:
    """Build a summary string from header metadata + truncated raw text."""
    parts = [f"Section: {section.title}"]
    parts.append(f"Path: {' > '.join(section.header_path)}")

    if section.children:
        child_titles = [tree.sections[c].title for c in section.children]
        parts.append(f"Subsections: {', '.join(child_titles)}")

    raw = PAGE_MARKER.sub("", raw)
    raw = IMAGE_TAG.sub("", raw).strip()
    if raw:
        truncated = raw[:max_chars]
        if len(raw) > max_chars:
            truncated = truncated.rsplit(" ", 1)[0] + "..."
        parts.append(truncated)

    return "\n".join(parts)


def create_section_summaries_heuristic(
    tree: DocumentTree,
    section_texts: dict[str, str],
    max_chars: int,
) -> list[TextNode]:
    """Build summary TextNodes using truncated text (no LLM)."""
    nodes: list[TextNode] = []
    for section_id, section in tree.sections.items():
        raw = section_texts.get(section_id, "").strip()
        summary = heuristic_summary(section, tree, raw, max_chars)
        nodes.append(make_summary_node(section, summary))
    return nodes


# ══════════════════════════════════════════════════════════════════════════
# HEURISTIC ENTITY GRAPH
# ══════════════════════════════════════════════════════════════════════════


def build_entity_graph(
    tree: DocumentTree,
    section_texts: dict[str, str],
) -> None:
    """Build entity graph from section titles and cross-references.

    Phase 1: each section title becomes an entity.
    Phase 2: scan section text for mentions of other section titles.
    Phase 3: extract acronyms and link to containing sections.
    """
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

    for sid, text in section_texts.items():
        text_lower = text.lower()
        for canonical, entity in tree.entities.items():
            if len(canonical) < 4:
                continue
            if canonical in text_lower and sid not in entity.section_ids:
                entity.section_ids.append(sid)

    for sid, text in section_texts.items():
        for match in ACRONYM_RE.finditer(text):
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
