import os
import pickle
import argparse
import json
import shutil
from time import strftime

from src.knowledge_graph.models import Chunk
from src.knowledge_graph.extractors import (
    JsonExtractor,
    KeyBERTExtractor,
    OpenRouterExtractor,
    SLMExtractor,
    BaseExtractor,
)
from src.knowledge_graph.models import KGPipelineConfig

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

CHUNKS_PKL = os.path.join(
    PROJECT_ROOT, "index", "sections", "textbook_index_chunks.pkl"
)
META_PKL = os.path.join(
    PROJECT_ROOT, "index", "sections", "textbook_index_meta.pkl"
)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "knowledge_graph")
RUNS_DIR = os.path.join(OUTPUT_DIR, "runs")
EXTRACTIONS_DIR = os.path.join(OUTPUT_DIR, "extractions")
LATEST_EXTRACTIONS = os.path.join(EXTRACTIONS_DIR, "latest.json")


def get_latest_extractions_path() -> str:
    """Return the real path behind the ``extractions/latest.json`` symlink.

    Raises:
        FileNotFoundError: If no extraction has been written yet.
    """
    if not os.path.exists(LATEST_EXTRACTIONS):
        raise FileNotFoundError(
            "No extractions found. Run llm_extract_keywords.py first, or pass "
            "--extractions <path> to point at an existing file."
        )
    return os.path.realpath(LATEST_EXTRACTIONS)


RUN_TIMESTAMP_FORMAT = "%Y-%m-%d_%H-%M-%S"


def create_run_dir() -> str:
    """Create a timestamped run directory and return its path."""
    run_dir = os.path.join(RUNS_DIR, strftime(RUN_TIMESTAMP_FORMAT))
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def setup_input_dir(run_dir: str, extractions_path: str | None) -> None:
    """Create input/ with symlinks to pkl sources.

    If *extractions_path* is provided (JsonExtractor was selected), a full copy
    of that file is placed in input/ for reproducibility.  For live extractors
    the extraction happens inside ``build_kg`` and is not cached here.
    """
    input_dir = os.path.join(run_dir, "input")
    os.makedirs(input_dir, exist_ok=True)

    os.symlink(os.path.abspath(CHUNKS_PKL), os.path.join(input_dir, "chunks.pkl"))
    os.symlink(os.path.abspath(META_PKL), os.path.join(input_dir, "meta.pkl"))

    if extractions_path is not None:
        shutil.copy2(extractions_path, os.path.join(input_dir, "extractions.json"))


def write_config(
    run_dir: str,
    cfg: KGPipelineConfig,
    extractor_config: dict,
    extractions_path: str | None,
) -> None:
    config = {
        "extractor": extractor_config,
        "linker": {
            "class": "CooccurrenceLinker",
            "min_cooccurrence": cfg.min_cooccurrence,
        },
        "chunks_pkl": CHUNKS_PKL,
        "meta_pkl": META_PKL,
        "timestamp": os.path.basename(run_dir),
    }
    if extractions_path is not None:
        config["extractions_path"] = extractions_path
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def update_latest_symlink(run_dir: str) -> None:
    latest = os.path.join(RUNS_DIR, "latest")
    if os.path.islink(latest):
        os.unlink(latest)
    os.symlink(os.path.abspath(run_dir), latest)


def build_extractor(args: argparse.Namespace, cfg: KGPipelineConfig) -> tuple[BaseExtractor, dict]:
    """Instantiate and return the chosen extractor plus its resolved config dict.

    Returns:
        (extractor, extractor_config, extractions_path)
        *extractions_path* is the JSON file used (JsonExtractor only), else None.
    """
    if args.extractor == "json":
        path = args.extractions or get_latest_extractions_path()
        extractor = JsonExtractor(input_path=path)
        return extractor, {"class": "JsonExtractor", "input_path": path}

    if args.extractor == "openrouter":
        api_key = args.api_key or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenRouter API key required. Pass --api_key or set OPENROUTER_API_KEY."
            )
        extractor = OpenRouterExtractor(
            api_key=api_key,
            model=args.model,
            top_n=args.top_n or cfg.top_n,
            adaptive_top_n=args.adaptive_top_n,
        )
        return (
            extractor,
            {
                "class": "OpenRouterExtractor",
                "model": args.model,
                "top_n": args.top_n or cfg.top_n,
                "adaptive_top_n": args.adaptive_top_n,
            },
        )

    if args.extractor == "keybert":
        extractor = KeyBERTExtractor(
            model=args.keybert_model,
            top_n=args.top_n or cfg.top_n,
        )
        return (
            extractor,
            {
                "class": "KeyBERTExtractor",
                "model": args.keybert_model,
                "top_n": args.top_n or cfg.top_n,
            },
        )

    if args.extractor == "slm":
        extractor = SLMExtractor(
            model_path=args.slm_model_path,
            n_threads=args.slm_threads,
            top_n=args.top_n or cfg.top_n,
        )
        return (
            extractor,
            {
                "class": "SLMExtractor",
                "model_path": args.slm_model_path,
                "n_threads": args.slm_threads,
                "top_n": args.top_n or cfg.top_n,
            },
        )

    raise ValueError(f"Unknown extractor: {args.extractor!r}")


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
