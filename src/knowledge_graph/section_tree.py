from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Optional

import networkx as nx

from src.knowledge_graph.models import Chunk
from src.knowledge_graph.utils import (
    HEADING_PATTERN,
    KW_PATTERN,
    Normalizer,
    extract_ngrams,
)

_NUMBER_RE = re.compile(r"(\d+(?:\.\d+)*)")

# Tokens to strip from heading text before building heading_keywords
_HEADING_PREFIX_RE = re.compile(r"\b(section|chapter)\b", re.IGNORECASE)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _extract_section_number(heading: str) -> str | None:
    """Return the section number from a heading like 'Section 13.1 ...'."""
    m = _NUMBER_RE.search(heading)
    return m.group(1) if m else None


def _parent_number(number: str) -> str | None:
    """Return the parent section number, or None for a top-level number."""
    parts = number.split(".")
    return ".".join(parts[:-1]) if len(parts) > 1 else None


def _build_heading_keywords(heading: str) -> set[str]:
    """Tokenize a section heading into a normalized keyword set.

    Strips the section number and "Section"/"Chapter" prefixes, then
    produces normalized unigrams, bigrams, and trigrams from the
    remaining words — matching the n-gram strategy used for KG nodes.
    """
    text = _NUMBER_RE.sub("", heading)
    text = _HEADING_PREFIX_RE.sub("", text).strip()
    return extract_ngrams(text, HEADING_PATTERN)


def _tokenize_query(query: str) -> set[str]:
    """Extract normalized unigrams, bigrams, and trigrams from a raw query.

    Unlike ``extract_query_nodes``, this does **not** filter against the KG
    graph — all normalized query tokens are returned.
    """
    return extract_ngrams(query, KW_PATTERN)


# ── Data model ────────────────────────────────────────────────────────────────


@dataclass
class SectionNode:
    heading: str  # e.g. "Section 13.1 Physical Storage Media"
    level: int  # 1 = chapter, 2 = section, 3 = subsection
    chapter: int  # e.g. 13
    section_number: str  # e.g. "13.1"
    chunk_ids: list[int] = field(default_factory=list)
    keyword_set: set[str] = field(default_factory=set)
    children: list[SectionNode] = field(default_factory=list)
    parent: Optional[SectionNode] = field(default=None, repr=False, compare=False)
    heading_keywords: set[str] = field(default_factory=set, repr=False, compare=False)


class SectionTree:
    """Tree mirroring the textbook's heading hierarchy with aggregated KG keywords."""

    def __init__(self, root: SectionNode) -> None:
        self.root = root
        self.node_index: dict[str, SectionNode] = {}  # heading → node
        self._number_index: dict[str, SectionNode] = {}  # section_number → node
        self.chunk_to_sections: dict[
            int, list[SectionNode]
        ] = {}  # chunk_id → leaf nodes

    # ── Index helpers ─────────────────────────────────────────────────────────

    def _register(self, node: SectionNode) -> None:
        self.node_index[node.heading] = node
        self._number_index[node.section_number] = node

    def get_nodes_at_level(self, level: int) -> list[SectionNode]:
        return [n for n in self.node_index.values() if n.level == level]

    # ── Query-time scoring ────────────────────────────────────────────────────

    def _score_section_kg(
        self,
        node: SectionNode,
        query_keywords: set[str],
        alpha: float = 0.6,
    ) -> float:
        """KG keyword overlap score: coverage × alpha + specificity × (1 - alpha).

        Coverage:    fraction of query keywords present in the section.
        Specificity: fraction of the section's keywords that are query keywords.
        """
        if not node.keyword_set or not query_keywords:
            return 0.0
        matched = query_keywords & node.keyword_set
        if not matched:
            return 0.0
        coverage = len(matched) / len(query_keywords)
        specificity = len(matched) / len(node.keyword_set)
        return alpha * coverage + (1 - alpha) * specificity

    def _score_section_heading(
        self,
        node: SectionNode,
        query_tokens: set[str],
        alpha: float = 0.6,
    ) -> float:
        """Heading keyword overlap score: coverage × alpha + specificity × (1 - alpha).

        Matches independently-tokenized query tokens against the pre-built
        heading keyword set.  Uses the same formula as ``_score_section_kg``
        for a consistent scale.
        """
        if not node.heading_keywords or not query_tokens:
            return 0.0
        matched = query_tokens & node.heading_keywords
        if not matched:
            return 0.0
        coverage = len(matched) / len(query_tokens)
        specificity = len(matched) / len(node.heading_keywords)
        return alpha * coverage + (1 - alpha) * specificity

    def get_all_descendant_chunk_ids(self, node: SectionNode) -> list[int]:
        ids: list[int] = list(node.chunk_ids)
        for child in node.children:
            ids.extend(self.get_all_descendant_chunk_ids(child))
        return ids

    def get_chunk_scores(
        self,
        query_keywords: set[str],
        query: str | None = None,
        heading_alpha: float = 0.5,
        inheritance_decay: float = 0.5,
        alpha: float = 0.6,
    ) -> dict[int, float]:
        """Return chunk_id → normalized section-relevance score.

        Hybrid scoring blends two independent signals per section node:

        - **Heading keyword match** (structural): overlap between
          independently-tokenized query tokens and the pre-built heading
          keyword set.  Captures queries phrased differently from the KG
          vocabulary; independent of which terms exist as KG nodes.
        - **KG keyword overlap** (lexical): coverage × alpha + specificity ×
          (1 - alpha) using the node's aggregated KG keyword set.

        ``heading_alpha`` controls the blend (1.0 = heading-only, 0.0 =
        KG-only).  Falls back to KG-only when ``query`` is None or heading
        keywords are absent.

        **Top-down inheritance** propagates a parent's score to its children:

            effective(node) = own_score(node) + inheritance_decay × effective(parent)

        This ensures that if section 13.1 is highly relevant, its subsections
        13.1.1, 13.1.2, … receive a proportional boost even if they score
        lower on their own.  Each chunk gets the effective score of its direct
        section node; chunks in more specific subsections that also match are
        doubly reinforced.

        Final scores are normalized to [0, 1].
        """
        if not self.node_index:
            return {}

        # Tokenize raw query independently for heading matching
        query_tokens: set[str] = set()
        if query is not None:
            normalizer = Normalizer()
            query_tokens = _tokenize_query(query)

        # ── Step 1: Compute own score for every node ──────────────────────────
        own_scores: dict[str, float] = {}
        for heading, node in self.node_index.items():
            kg_score = self._score_section_kg(node, query_keywords, alpha)

            if query_tokens and node.heading_keywords:
                heading_score = self._score_section_heading(node, query_tokens, alpha)
                own_scores[heading] = (
                    heading_alpha * heading_score + (1 - heading_alpha) * kg_score
                )
            else:
                own_scores[heading] = kg_score

        # ── Step 2: Top-down DFS — effective = own + decay × parent_effective ─
        effective: dict[str, float] = {}

        def _propagate(node: SectionNode, parent_eff: float) -> None:
            own = own_scores.get(node.heading, 0.0)
            eff = own + inheritance_decay * parent_eff
            effective[node.heading] = eff
            for child in node.children:
                _propagate(child, eff)

        for top_level in self.root.children:
            _propagate(top_level, 0.0)

        # ── Step 3: Assign chunk scores from their direct section node ────────
        chunk_scores: dict[int, float] = {}
        for heading, node in self.node_index.items():
            eff = effective.get(heading, 0.0)
            if eff <= 0.0:
                continue
            for chunk_id in node.chunk_ids:
                chunk_scores[chunk_id] = max(chunk_scores.get(chunk_id, 0.0), eff)

        if not chunk_scores:
            return {}

        max_score = max(chunk_scores.values())
        if max_score > 0:
            chunk_scores = {cid: s / max_score for cid, s in chunk_scores.items()}
        return chunk_scores

    # ── Serialization ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        def node_to_dict(n: SectionNode) -> dict:
            return {
                "heading": n.heading,
                "level": n.level,
                "chapter": n.chapter,
                "section_number": n.section_number,
                "chunk_ids": n.chunk_ids,
                "keyword_set": sorted(n.keyword_set),
                "heading_keywords": sorted(n.heading_keywords),
                "children": [node_to_dict(c) for c in n.children],
            }

        return node_to_dict(self.root)

    @classmethod
    def from_dict(cls, data: dict) -> SectionTree:
        def dict_to_node(d: dict, parent: SectionNode | None) -> SectionNode:
            node = SectionNode(
                heading=d["heading"],
                level=d["level"],
                chapter=d["chapter"],
                section_number=d["section_number"],
                chunk_ids=d["chunk_ids"],
                keyword_set=set(d["keyword_set"]),
                heading_keywords=set(d.get("heading_keywords", [])),
                parent=parent,
            )
            node.children = [dict_to_node(c, node) for c in d.get("children", [])]
            return node

        root = dict_to_node(data, None)
        tree = cls(root)
        tree._rebuild_indexes(root)
        return tree

    def _rebuild_indexes(self, node: SectionNode) -> None:
        if node.heading != "root":
            self._register(node)
            for chunk_id in node.chunk_ids:
                self.chunk_to_sections.setdefault(chunk_id, []).append(node)
        for child in node.children:
            self._rebuild_indexes(child)


# ── Build ─────────────────────────────────────────────────────────────────────


def build_section_tree(
    chunks: list[Chunk],
    graph: nx.Graph,
) -> SectionTree:
    """Build a SectionTree from KG chunks and a populated knowledge graph.

    Steps:
    1. Collect unique sections from chunk metadata (heading, level, chapter).
    2. Attach each section node to its parent using the section number prefix
       (e.g. "13.1" → parent "13").
    3. Assign chunk_ids to their leaf section nodes.
    4. Populate leaf keyword_sets from the graph's ``chunk_ids`` node attributes.
    5. Aggregate keyword sets bottom-up so every ancestor contains the union of
       all descendant keywords.
    6. Extract heading_keywords for each section using the Normalizer.

    Args:
        chunks: Chunk objects with a ``section`` metadata field containing the
                immediate heading string, e.g. ``"Section 1.1 Foo Bar"``
                (produced by ``index_builder.build_index``).  ``level`` and
                ``chapter`` are derived from the section number via regex.
        graph:  NetworkX graph from the KG pipeline; each node has a
                ``chunk_ids`` attribute listing which chunks contain it.

    Returns:
        A fully populated ``SectionTree``.
    """
    root = SectionNode(heading="root", level=0, chapter=0, section_number="")
    tree = SectionTree(root)

    # ── Step 1: Collect unique sections ──────────────────────────────────────
    seen: dict[str, SectionNode] = {}  # section_number → SectionNode
    for chunk in chunks:
        meta = chunk.metadata
        heading = meta.get("section", "")
        if not heading:
            continue
        section_number = _extract_section_number(heading)
        if section_number is None:
            continue
        if section_number not in seen:
            level = section_number.count(".") + 1
            chapter = int(section_number.split(".")[0])
            seen[section_number] = SectionNode(
                heading=heading,
                level=level,
                chapter=chapter,
                section_number=section_number,
            )

    # ── Step 2: Build tree structure (shortest numbers first = parents first) ─
    for section_number, node in sorted(
        seen.items(), key=lambda x: (x[0].count("."), x[0])
    ):
        parent_num = _parent_number(section_number)
        parent_node = seen.get(parent_num, root) if parent_num else root
        node.parent = parent_node
        parent_node.children.append(node)
        tree._register(node)

    # ── Step 3: Assign chunk_ids to leaf nodes ────────────────────────────────
    for chunk in chunks:
        meta = chunk.metadata
        section_number = _extract_section_number(meta.get("section", ""))
        if not section_number or section_number not in seen:
            continue
        leaf = seen[section_number]
        if chunk.id not in leaf.chunk_ids:
            leaf.chunk_ids.append(chunk.id)
        tree.chunk_to_sections.setdefault(chunk.id, []).append(leaf)

    # ── Step 4: Populate keyword sets from KG graph ───────────────────────────
    for kg_node_name, kg_node_data in graph.nodes(data=True):
        for chunk_id in kg_node_data.get("chunk_ids", []):
            for leaf in tree.chunk_to_sections.get(chunk_id, []):
                leaf.keyword_set.add(kg_node_name)

    # ── Step 5: Bottom-up keyword aggregation ─────────────────────────────────
    def _aggregate(node: SectionNode) -> None:
        for child in node.children:
            _aggregate(child)
            node.keyword_set |= child.keyword_set

    _aggregate(root)

    # ── Step 6: Extract heading keywords for each section ─────────────────────
    normalizer = Normalizer()
    for node in seen.values():
        node.heading_keywords = _build_heading_keywords(node.heading)

    return tree


# ── Persist / load ────────────────────────────────────────────────────────────


def save_section_tree(tree: SectionTree, run_dir: str) -> str:
    """Serialize *tree* to ``section_tree.json`` inside *run_dir*.

    Returns:
        The full path of the written file.
    """
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, "section_tree.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(tree.to_dict(), f, indent=2, ensure_ascii=False)
    return path


def load_section_tree(run_dir: str) -> SectionTree:
    """Load the section tree from ``section_tree.json`` in *run_dir*.

    Raises:
        FileNotFoundError: If ``section_tree.json`` is not found.
    """
    path = os.path.join(run_dir, "section_tree.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No section_tree.json found in {run_dir!r}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return SectionTree.from_dict(data)
