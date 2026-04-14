import argparse
import json
import logging
import os
import shutil
from time import strftime

from dotenv import load_dotenv

from src.knowledge_graph.build import (
    CHUNKS_PKL,
    JSON_KW_PATH,
    META_PKL,
    OUTPUT_DIR,
    PROJECT_ROOT,
)
from src.knowledge_graph.extractors import JsonExtractor
from src.knowledge_graph.linkers import CooccurrenceLinker
from src.knowledge_graph.pipeline import build_kg
from src.knowledge_graph.models import KGPipelineConfig


logger = logging.getLogger(__name__)

_RUN_TIMESTAMP_FORMAT = "%Y-%m-%d_%H-%M-%S"


def _create_run_dir(runs_dir: str) -> str:
    """Create a timestamped run directory and return its path."""
    run_dir = os.path.join(runs_dir, strftime(_RUN_TIMESTAMP_FORMAT))
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def _setup_input_dir(run_dir: str) -> None:
    """Create input/ with symlinks to pkl sources and a copy of the extractions JSON."""
    input_dir = os.path.join(run_dir, "input")
    os.makedirs(input_dir, exist_ok=True)

    # Symlinks for the (large) pkl files — no copy
    os.symlink(os.path.abspath(CHUNKS_PKL),
               os.path.join(input_dir, "chunks.pkl"))
    os.symlink(os.path.abspath(META_PKL), os.path.join(input_dir, "meta.pkl"))

    # Full copy of the keyword extractions JSON
    shutil.copy2(JSON_KW_PATH, os.path.join(input_dir, "extractions.json"))


def _write_config(run_dir: str, cfg: KGPipelineConfig) -> None:
    config = {
        "extractor": {"class": "JsonExtractor", "input_path": JSON_KW_PATH},
        "linker": {"class": "CooccurrenceLinker", "min_cooccurrence": cfg.min_cooccurrence},
        "chunks_pkl": CHUNKS_PKL,
        "meta_pkl": META_PKL,
        "top_n": cfg.top_n,
        "timestamp": os.path.basename(run_dir),
    }
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def _update_latest_symlink(runs_dir: str, run_dir: str) -> None:
    latest = os.path.join(runs_dir, "latest")
    if os.path.islink(latest):
        os.unlink(latest)
    os.symlink(os.path.abspath(run_dir), latest)


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
    args = parser.parse_args()

    cfg = KGPipelineConfig.from_yaml(args.config)
    logger.info("Loaded config from %s", args.config)

    runs_dir = os.path.join(OUTPUT_DIR, "runs")
    run_dir = _create_run_dir(runs_dir)
    logger.info("Run directory: %s", run_dir)

    _setup_input_dir(run_dir)
    _write_config(run_dir, cfg)

    extractor = JsonExtractor(input_path=JSON_KW_PATH)
    linker = CooccurrenceLinker(min_cooccurrence=cfg.min_cooccurrence)

    build_kg(output_dir=run_dir, extractor=extractor, linker=linker)

    _update_latest_symlink(runs_dir, run_dir)
    logger.info("Updated: %s -> %s", os.path.join(runs_dir, "latest"), run_dir)


if __name__ == "__main__":
    load_dotenv()  # Load environment variables from .env if present
    main()
