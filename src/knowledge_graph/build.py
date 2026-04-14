import os
import pickle

from src.knowledge_graph.models import Chunk

# ---------------------------------------------------------------------------
# Project-level path constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

CHUNKS_PKL = os.path.join(
    PROJECT_ROOT, "index", "sections", "textbook_index_chunks.pkl"
)
META_PKL = os.path.join(
    PROJECT_ROOT, "index", "sections", "textbook_index_meta.pkl"
)
# TODO: Update this path to point to the actual extractions JSON after running the extractor
JSON_KW_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "knowledge_graph",
    "all__google_gemini-3-flash-preview__extractions__2.json",
)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "knowledge_graph")
RUNS_DIR = os.path.join(OUTPUT_DIR, "runs")

# ---------------------------------------------------------------------------
# Chunk loader (build-time: reads pickle files from index_builder)
# ---------------------------------------------------------------------------


def load_chunks(
    chunks_path: str,
    meta_path: str,
    chapter_filter: str | None = None,
    exclude_chapters: list[str] | None = None,
    chunk_ids: list[int] | None = None,
) -> list[Chunk]:
    """Load pre-chunked text and metadata from pickle files into Chunk objects.

    Args:
        chunks_path:      Path to ``*_chunks.pkl`` produced by ``index_builder``.
        meta_path:        Path to ``*_meta.pkl`` produced by ``index_builder``.
        chapter_filter:   If set, only include chunks whose ``section_path``
                          starts with this prefix (e.g. ``"Chapter 3 "``).
        exclude_chapters: Skip chunks whose ``section_path`` starts with any
                          of these prefixes.
        chunk_ids:        If set, only include chunks with these IDs.

    Returns:
        List of ``Chunk`` objects with ``id``, ``text``, and ``metadata``.

    Raises:
        ValueError: If the number of chunks and metadata entries differ.
    """
    with open(chunks_path, "rb") as f:
        texts: list[str] = pickle.load(f)

    with open(meta_path, "rb") as f:
        metas: list[dict] = pickle.load(f)

    if len(texts) != len(metas):
        raise ValueError(
            f"Mismatch: {len(texts)} chunks vs {len(metas)} metadata entries"
        )

    filtering = chapter_filter or exclude_chapters or chunk_ids is not None
    if not filtering:
        return [
            Chunk(id=meta.get("chunk_id", i), text=text, metadata=meta)
            for i, (text, meta) in enumerate(zip(texts, metas))
        ]

    chunks: list[Chunk] = []
    for i, (text, meta) in enumerate(zip(texts, metas)):
        chunk_id = meta.get("chunk_id", i)
        section_path = meta.get("section_path", "")

        if exclude_chapters and any(
            section_path.startswith(ex) for ex in exclude_chapters
        ):
            continue
        if chunk_ids is not None and chunk_id not in chunk_ids:
            continue
        if chapter_filter and not section_path.startswith(chapter_filter):
            continue

        chunks.append(Chunk(id=chunk_id, text=text, metadata=meta))
    return chunks
