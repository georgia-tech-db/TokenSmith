import argparse
import logging
import os

from dotenv import load_dotenv

from src.knowledge_graph.build import (
    RUNS_DIR,
    PROJECT_ROOT,
    build_extractor,
    create_run_dir,
    setup_input_dir,
    write_config,
    update_latest_symlink,
)

from src.knowledge_graph.linkers import CooccurrenceLinker
from src.knowledge_graph.pipeline import build_kg
from src.knowledge_graph.models import KGPipelineConfig


logger = logging.getLogger(__name__)


_EXTRACTOR_CHOICES = ("json", "openrouter", "keybert", "slm")


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

    extractor, extractor_config = build_extractor(args, cfg)
    logger.info("Using extractor: %s", extractor_config["class"])

    run_dir = create_run_dir()
    logger.info("Run directory: %s", run_dir)

    extractions_path = (
        extractor_config.get("input_path")
        if extractor_config["class"] == "JsonExtractor"
        else None
    )
    setup_input_dir(run_dir, extractions_path)
    write_config(run_dir, cfg, extractor_config, extractions_path)

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

    update_latest_symlink(run_dir)
    logger.info("Updated: %s -> %s", os.path.join(RUNS_DIR, "latest"), run_dir)


if __name__ == "__main__":
    load_dotenv()
    main()
