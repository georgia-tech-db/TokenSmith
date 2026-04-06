"""Document tree and entity graph for BookRAG-style hierarchical retrieval.

Structures:
  - SectionNode: one node in the document hierarchy (chapter / section / subsection)
  - Relation:    typed edge between two entities, grounded to a section
  - EntityNode:  a concept with aliases, type, provenance, and relations
  - DocumentTree: full tree + entity graph, with subtree / co-occurrence queries
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SectionNode:
    id: str
    title: str
    depth: int  # 1=chapter, 2=section, 3=subsection, …
    parent_id: str | None
    children: list[str] = field(default_factory=list)
    leaf_ids: list[str] = field(default_factory=list)
    summary: str = ""
    header_path: list[str] = field(default_factory=list)
    source: str = ""


@dataclass
class Relation:
    """Typed, provenance-grounded edge between two entities."""
    source: str          # canonical key of the source entity
    target: str          # canonical key of the target entity
    relation_type: str   # is_a | part_of | uses | implements | extends | …
    section_id: str      # section where this relation was observed


@dataclass
class EntityNode:
    name: str
    canonical: str                                       # lowercased canonical key
    section_ids: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)     # alternative names / abbreviations
    entity_type: str = ""                                # concept | algorithm | data_structure | …
    provenance: dict[str, float] = field(default_factory=dict)   # section_id → salience (0–1)
    relations: list[Relation] = field(default_factory=list)


class DocumentTree:
    """Hierarchical document tree with entity graph linking."""

    def __init__(self) -> None:
        self.sections: dict[str, SectionNode] = {}
        self.entities: dict[str, EntityNode] = {}
        self.root_ids: list[str] = []

    # ── tree mutation ─────────────────────────────────────────────────────

    def add_section(self, section: SectionNode) -> None:
        self.sections[section.id] = section
        if section.parent_id is None:
            self.root_ids.append(section.id)
        elif section.parent_id in self.sections:
            parent = self.sections[section.parent_id]
            if section.id not in parent.children:
                parent.children.append(section.id)

    # ── tree queries ──────────────────────────────────────────────────────

    def subtree_leaf_ids(self, section_id: str) -> list[str]:
        """All leaf node IDs reachable from this section (recursive)."""
        section = self.sections[section_id]
        ids = list(section.leaf_ids)
        for child_id in section.children:
            ids.extend(self.subtree_leaf_ids(child_id))
        return ids

    def subtree_section_ids(self, section_id: str) -> list[str]:
        """All section IDs in a subtree (inclusive)."""
        ids = [section_id]
        for child_id in self.sections[section_id].children:
            ids.extend(self.subtree_section_ids(child_id))
        return ids

    # ── entity queries ────────────────────────────────────────────────────

    def sections_for_entity(self, entity_canonical: str) -> list[str]:
        entity = self.entities.get(entity_canonical)
        if entity is None:
            return []
        return list(entity.section_ids)

    def entities_in_section(self, section_id: str) -> set[str]:
        return {
            e.canonical
            for e in self.entities.values()
            if section_id in e.section_ids
        }

    def cooccurring_entities(self, entity_canonical: str) -> set[str]:
        """Entities that share at least one section with the given entity."""
        sections = self.sections_for_entity(entity_canonical)
        cooccurring: set[str] = set()
        for sid in sections:
            cooccurring |= self.entities_in_section(sid)
        cooccurring.discard(entity_canonical)
        return cooccurring

    def resolve_alias(self, name: str) -> EntityNode | None:
        """Look up an entity by canonical key *or* any alias."""
        key = name.lower().strip()
        if key in self.entities:
            return self.entities[key]
        for entity in self.entities.values():
            if key in (a.lower() for a in entity.aliases):
                return entity
        return None

    def entity_relations(self, entity_canonical: str) -> list[Relation]:
        """All relations where this entity is source or target."""
        entity = self.entities.get(entity_canonical)
        if entity is None:
            return []
        rels = list(entity.relations)
        for other in self.entities.values():
            if other.canonical == entity_canonical:
                continue
            for r in other.relations:
                if r.target == entity_canonical:
                    rels.append(r)
        return rels

    # ── persistence ───────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        data = {
            "root_ids": self.root_ids,
            "sections": {
                sid: {
                    "id": s.id,
                    "title": s.title,
                    "depth": s.depth,
                    "parent_id": s.parent_id,
                    "children": s.children,
                    "leaf_ids": s.leaf_ids,
                    "summary": s.summary,
                    "header_path": s.header_path,
                    "source": s.source,
                }
                for sid, s in self.sections.items()
            },
            "entities": {
                name: {
                    "name": e.name,
                    "canonical": e.canonical,
                    "section_ids": e.section_ids,
                    "aliases": e.aliases,
                    "entity_type": e.entity_type,
                    "provenance": e.provenance,
                    "relations": [
                        {
                            "source": r.source,
                            "target": r.target,
                            "relation_type": r.relation_type,
                            "section_id": r.section_id,
                        }
                        for r in e.relations
                    ],
                }
                for name, e in self.entities.items()
            },
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> DocumentTree:
        with open(path) as f:
            data = json.load(f)
        tree = cls()
        tree.root_ids = data["root_ids"]

        for sid, sd in data["sections"].items():
            tree.sections[sid] = SectionNode(
                id=sd["id"],
                title=sd["title"],
                depth=sd["depth"],
                parent_id=sd["parent_id"],
                children=sd["children"],
                leaf_ids=sd["leaf_ids"],
                summary=sd["summary"],
                header_path=sd["header_path"],
                source=sd["source"],
            )

        for name, ed in data["entities"].items():
            tree.entities[name] = EntityNode(
                name=ed["name"],
                canonical=ed["canonical"],
                section_ids=ed["section_ids"],
                aliases=ed.get("aliases", []),
                entity_type=ed.get("entity_type", ""),
                provenance=ed.get("provenance", {}),
                relations=[
                    Relation(**r) for r in ed.get("relations", [])
                ],
            )

        return tree
