import json
import os

import faiss
import networkx as nx
import numpy as np

from src.knowledge_graph.build import RUNS_DIR  # re-exported for callers  # noqa: F401
from src.knowledge_graph.section_tree import SectionTree, load_section_tree
from src.knowledge_graph.summary_tree import SummaryEntry, load_summary_index as _load_summary_index


def load_graph(path: str) -> nx.Graph:
    """Load a NetworkX graph from a ``graph.json`` node-link file."""
    with open(path, "r", encoding="utf-8") as f:
        return nx.node_link_graph(json.load(f))


def load_run_chunks(path: str) -> dict[int, str]:
    """Load chunk text from a ``chunks.json`` run artifact.

    JSON object keys must be strings, so the file stores chunk IDs as strings.
    This function converts them back to ``int``.

    Returns a mapping of integer chunk ID → text.
    """
    with open(path, "r", encoding="utf-8") as f:
        return {int(k): v for k, v in json.load(f).items()}


def resolve_run_dir(path: str) -> str:
    """Return the concrete run directory to load from.

    - If ``path/graph.json`` exists, *path* is already a run directory.
    - If ``path/latest`` is a symlink, resolve and return it.
    - Otherwise raise ``FileNotFoundError``.
    """
    if os.path.isfile(os.path.join(path, "graph.json")):
        return path
    latest = os.path.join(path, "latest")
    if os.path.islink(latest):
        resolved = os.path.realpath(latest)
        if os.path.isfile(os.path.join(resolved, "graph.json")):
            return resolved
    raise FileNotFoundError(
        f"Cannot resolve run dir from {path!r}: "
        "no graph.json found and no valid 'latest' symlink."
    )


def load_graph_and_chunks(output_dir: str) -> tuple[nx.Graph, dict[int, str]]:
    """Load the most recently persisted graph and chunks from *output_dir*.

    Accepts either a specific run directory (containing ``graph.json`` and
    ``chunks.json``) or a parent ``runs/`` directory with a ``latest`` symlink.

    Returns:
        ``(graph, chunks)`` where *chunks* maps ``int`` chunk IDs to text.

    Raises:
        FileNotFoundError: If the run directory cannot be resolved.
    """
    run_dir = resolve_run_dir(output_dir)
    graph = load_graph(os.path.join(run_dir, "graph.json"))
    chunks = load_run_chunks(os.path.join(run_dir, "chunks.json"))
    return graph, chunks


def load_graph_chunks_and_tree(
    output_dir: str,
) -> tuple[nx.Graph, dict[int, str], SectionTree | None]:
    """Like ``load_graph_and_chunks`` but also loads the section tree.

    Returns:
        ``(graph, chunks, section_tree)`` — *section_tree* is ``None`` when
        ``section_tree.json`` is not present, so callers fall back gracefully
        to node-only scoring.
    """
    run_dir = resolve_run_dir(output_dir)
    graph, chunks = load_graph_and_chunks(run_dir)
    try:
        tree = load_section_tree(run_dir)
    except FileNotFoundError:
        tree = None
    return graph, chunks, tree


def load_summary_data(
    output_dir: str,
) -> tuple[faiss.Index, list[SummaryEntry]] | tuple[None, None]:
    """Load the summary FAISS index and metadata from *output_dir*.

    Returns:
        ``(index, entries)`` when both artifacts exist, ``(None, None)`` otherwise.
    """
    try:
        run_dir = resolve_run_dir(output_dir)
        return _load_summary_index(run_dir)
    except FileNotFoundError:
        return None, None


def load_canonicalization_data(
    run_dir: str,
) -> tuple[dict[str, str], list[str], np.ndarray] | tuple[None, None, None]:
    """Load synonym table, canonical keywords, and embeddings from a run directory.

    Returns ``(None, None, None)`` when canonicalization artifacts are absent.
    """
    synonym_path = os.path.join(run_dir, "synonym_table.json")
    keywords_path = os.path.join(run_dir, "canonical_keywords.json")
    embeddings_path = os.path.join(run_dir, "canonical_embeddings.npy")

    if not all(os.path.exists(p) for p in [synonym_path, keywords_path, embeddings_path]):
        return None, None, None

    with open(synonym_path, "r", encoding="utf-8") as f:
        synonym_table: dict[str, str] = json.load(f)
    with open(keywords_path, "r", encoding="utf-8") as f:
        canonical_keywords: list[str] = json.load(f)
    canonical_embeddings = np.load(embeddings_path)

    return synonym_table, canonical_keywords, canonical_embeddings
