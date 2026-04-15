import argparse
import json
import logging
import os
import shutil
from time import strftime

from dotenv import load_dotenv

from src.knowledge_graph.build import (
    CHUNKS_PKL,
    META_PKL,
    RUNS_DIR,
    PROJECT_ROOT,
    get_latest_extractions_path,
)
from src.knowledge_graph.extractors import (
    JsonExtractor,
    KeyBERTExtractor,
    OpenRouterExtractor,
    SLMExtractor,
    BaseExtractor,
)
from src.knowledge_graph.linkers import CooccurrenceLinker
from src.knowledge_graph.pipeline import build_kg
from src.knowledge_graph.models import KGPipelineConfig


logger = logging.getLogger(__name__)

_RUN_TIMESTAMP_FORMAT = "%Y-%m-%d_%H-%M-%S"

_EXTRACTOR_CHOICES = ("json", "openrouter", "keybert", "slm")


def _create_run_dir() -> str:
    """Create a timestamped run directory and return its path."""
    run_dir = os.path.join(RUNS_DIR, strftime(_RUN_TIMESTAMP_FORMAT))
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def _setup_input_dir(run_dir: str, extractions_path: str | None) -> None:
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


def _write_config(
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


def _update_latest_symlink(run_dir: str) -> None:
    latest = os.path.join(RUNS_DIR, "latest")
    if os.path.islink(latest):
        os.unlink(latest)
    os.symlink(os.path.abspath(run_dir), latest)


def _build_extractor(args: argparse.Namespace, cfg: KGPipelineConfig) -> tuple[BaseExtractor, dict]:
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


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(description="Build the knowledge graph.")
    parser.add_argument(
        "--config",
        default=os.path.join(PROJECT_ROOT, "config", "config.yaml"),
        help="Path to project config YAML (default: config/config.yaml)",
    )

    # ── Extractor selection ────────────────────────────────────────────────
    parser.add_argument(
        "--extractor",
        choices=_EXTRACTOR_CHOICES,
        default="json",
        help="Keyword extractor to use (default: json)",
    )

    # json extractor
    parser.add_argument(
        "--extractions",
        default=None,
        metavar="PATH",
        help="[json] Path to a specific extractions JSON. "
        "Defaults to extractions/latest.json when --extractor json.",
    )

    # openrouter extractor
    parser.add_argument(
        "--api_key",
        default=None,
        help="[openrouter] OpenRouter API key (fallback: OPENROUTER_API_KEY env var)",
    )
    parser.add_argument(
        "--model",
        default="qwen/qwen3-next-80b-a3b-instruct",
        help="[openrouter] Model name (default: qwen/qwen3-next-80b-a3b-instruct)",
    )
    parser.add_argument(
        "--adaptive_top_n",
        action="store_true",
        default=False,
        help="[openrouter] Scale top_n as int(sqrt(len(chunk.text))) per chunk",
    )

    # keybert extractor
    parser.add_argument(
        "--keybert_model",
        default="all-MiniLM-L6-v2",
        help="[keybert] Sentence-transformer model name (default: all-MiniLM-L6-v2)",
    )

    # slm extractor
    parser.add_argument(
        "--slm_model_path",
        default="models/qwen2.5-1.5b-instruct-q5_k_m.gguf",
        help="[slm] Path to the GGUF model file",
    )
    parser.add_argument(
        "--slm_threads",
        type=int,
        default=8,
        help="[slm] Number of CPU threads (default: 8)",
    )

    # shared across live extractors
    parser.add_argument(
        "--top_n",
        type=int,
        default=None,
        help="[openrouter/keybert/slm] Keywords per chunk. Defaults to cfg.top_n.",
    )

    # ── Chunk filtering ───────────────────────────────────────────────────
    parser.add_argument(
        "--chapter",
        type=int,
        default=None,
        help="Only include chunks from this chapter number.",
    )
    parser.add_argument(
        "--exclude_chapters",
        type=int,
        nargs="+",
        default=[],
        help="Exclude chunks from these chapter numbers.",
    )

    args = parser.parse_args()

    cfg = KGPipelineConfig.from_yaml(args.config)
    logger.info("Loaded config from %s", args.config)

    extractor, extractor_config = _build_extractor(args, cfg)
    logger.info("Using extractor: %s", extractor_config["class"])

    run_dir = _create_run_dir()
    logger.info("Run directory: %s", run_dir)

    extractions_path = (
        extractor_config.get("input_path")
        if extractor_config["class"] == "JsonExtractor"
        else None
    )
    _setup_input_dir(run_dir, extractions_path)
    _write_config(run_dir, cfg, extractor_config, extractions_path)

    chapter_filter = f"Chapter {args.chapter} " if args.chapter else None
    exclude_chapters = [f"Chapter {c} " for c in args.exclude_chapters]

    linker = CooccurrenceLinker(min_cooccurrence=cfg.min_cooccurrence)
    build_kg(
        output_dir=run_dir,
        extractor=extractor,
        linker=linker,
        chapter_filter=chapter_filter,
        exclude_chapters=exclude_chapters or None,
    )

    _update_latest_symlink(run_dir)
    logger.info("Updated: %s -> %s", os.path.join(RUNS_DIR, "latest"), run_dir)


if __name__ == "__main__":
    load_dotenv()
    main()
