from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Callable

import faiss

from src.knowledge_graph.openrouter_client import OpenRouterClient
from src.knowledge_graph.section_tree import SectionNode, SectionTree
from src.knowledge_graph.prompts import (
    CHUNK_SUMMARY_PROMPT,
    SECTION_SUMMARY_PROMPT,
    SUMMARY_SYSTEM_PROMPT,
)

SUMMARY_INDEX_FILE = "summary_index.faiss"
SUMMARY_META_FILE = "summary_meta.json"


@dataclass
class SummaryEntry:
    section_number: str
    level: int  # 0 = chunk-group; matches SectionNode.level for section nodes
    chunk_ids: list[int]
    summary_text: str


def _windowed(items: list[int], window: int) -> list[list[int]]:
    """Split *items* into consecutive non-overlapping groups of size *window*."""
    return [items[i: i + window] for i in range(0, len(items), window)]


def _all_chunk_ids(node: SectionNode) -> list[int]:
    ids = list(node.chunk_ids)
    for child in node.children:
        ids.extend(_all_chunk_ids(child))
    return ids


def _collect_entries(
    node: SectionNode,
    chunks: dict[int, str],
    summarize_fn: Callable[[list[dict]], str],
    chunk_window: int,
    entries: list[SummaryEntry],
    section_summary_cache: dict[str, str],
) -> None:
    """Post-order DFS: build summaries bottom-up and populate *entries*."""
    if not node.children:
        # ── Leaf node ────────────────────────────────────────────────────────
        groups = _windowed(node.chunk_ids, chunk_window)
        group_summaries: list[str] = []

        for group in groups:
            text = "\n\n".join(chunks[cid] for cid in group if cid in chunks).strip()
            if not text:
                continue
            summary = summarize_fn([
                {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": CHUNK_SUMMARY_PROMPT.format(text=text)},
            ])
            entries.append(SummaryEntry(
                section_number=node.section_number,
                level=0,
                chunk_ids=list(group),
                summary_text=summary,
            ))
            group_summaries.append(summary)

        if group_summaries:
            section_summary = summarize_fn([
                {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": SECTION_SUMMARY_PROMPT.format(
                    heading=node.heading,
                    summaries="\n\n".join(group_summaries),
                )},
            ])
            entries.append(SummaryEntry(
                section_number=node.section_number,
                level=node.level,
                chunk_ids=list(node.chunk_ids),
                summary_text=section_summary,
            ))
            section_summary_cache[node.section_number] = section_summary
    else:
        # ── Internal node: recurse first ──────────────────────────────────
        for child in node.children:
            _collect_entries(
                child, chunks, summarize_fn, chunk_window, entries, section_summary_cache
            )

        child_summaries = [
            section_summary_cache[child.section_number]
            for child in node.children
            if child.section_number in section_summary_cache
        ]
        if child_summaries:
            section_summary = summarize_fn([
                {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": SECTION_SUMMARY_PROMPT.format(
                    heading=node.heading,
                    summaries="\n\n".join(child_summaries),
                )},
            ])
            entries.append(SummaryEntry(
                section_number=node.section_number,
                level=node.level,
                chunk_ids=_all_chunk_ids(node),
                summary_text=section_summary,
            ))
            section_summary_cache[node.section_number] = section_summary


def build_summary_index(
    client: OpenRouterClient,
    summary_model: str,
    section_tree: SectionTree,
    chunks: dict[int, str],
    embed_model: str,
    chunk_window: int,
    run_dir: str,
) -> tuple[faiss.Index, list[SummaryEntry]]:
    """Build LLM summaries for all tree levels and persist as a FAISS index.

    Args:
        section_tree:  Pre-built ``SectionTree`` for the corpus.
        chunks:        Mapping of chunk ID → raw text.

        embed_model:   SentenceTransformer model name for embedding summaries.
        chunk_window:  Number of adjacent chunks summarized together at the
                       leaf level (level=0).  Larger values → fewer LLM calls
                       but coarser granularity.
        run_dir:       Directory where ``summary_index.faiss`` and
                       ``summary_meta.json`` will be written.

    Returns:
        ``(index, entries)`` — the populated FAISS index and the parallel list
        of ``SummaryEntry`` objects (index position == FAISS row).
    """
    entries: list[SummaryEntry] = []
    section_summary_cache: dict[str, str] = {}
    def summarize_fn(messages): return client.chat(summary_model, messages)

    for top_level_node in section_tree.root.children:
        _collect_entries(
            top_level_node, chunks, summarize_fn, chunk_window, entries, section_summary_cache
        )

    if not entries:
        raise ValueError("No summaries generated — section tree may be empty.")

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(embed_model)
    texts = [e.summary_text for e in entries]
    embeddings = model.encode(texts, show_progress_bar=True).astype("float32")
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    os.makedirs(run_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(run_dir, SUMMARY_INDEX_FILE))
    with open(os.path.join(run_dir, SUMMARY_META_FILE), "w", encoding="utf-8") as f:
        json.dump([asdict(e) for e in entries], f, indent=2, ensure_ascii=False)

    return index, entries


def load_summary_index(run_dir: str) -> tuple[faiss.Index, list[SummaryEntry]]:
    """Load a persisted summary FAISS index and its metadata from *run_dir*.

    Raises:
        FileNotFoundError: If either artifact is absent.
    """
    index_path = os.path.join(run_dir, SUMMARY_INDEX_FILE)
    meta_path = os.path.join(run_dir, SUMMARY_META_FILE)

    if not os.path.isfile(index_path):
        raise FileNotFoundError(f"Summary index not found: {index_path}")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Summary metadata not found: {meta_path}")

    index = faiss.read_index(index_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        entries = [SummaryEntry(**d) for d in json.load(f)]

    return index, entries
