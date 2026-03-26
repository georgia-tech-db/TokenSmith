import argparse
import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from time import strftime

import yaml
from dotenv import load_dotenv

from src.knowledge_graph.build import (
    CHUNKS_PKL,
    JSON_KW_PATH,
    META_PKL,
    OUTPUT_DIR,
    PROJECT_ROOT,
    TOP_N,
    load_chunks,
)
from src.knowledge_graph.extractors import BaseExtractor, JsonExtractor
from src.knowledge_graph.linkers import CooccurrenceLinker
from src.knowledge_graph.persisters import NetworkxJsonPersister
from src.knowledge_graph.pipeline import Pipeline

logger = logging.getLogger(__name__)

_RUN_TIMESTAMP_FORMAT = "%Y-%m-%d_%H-%M-%S"


@dataclass
class KGPipelineConfig:
    corpus_description: str = ""
    min_cooccurrence: int = 0
    top_n: int = TOP_N

    @classmethod
    def from_yaml(cls, path: str) -> "KGPipelineConfig":
        """Load the ``kg_pipeline`` section from a project config YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        kg = dict(data.get("kg_pipeline", {}))
        return cls(**kg)


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

    logger.info("Loading chunks from:\n  %s\n  %s", CHUNKS_PKL, META_PKL)
    chunks = load_chunks(CHUNKS_PKL, META_PKL)
    logger.info("Loaded %d chunks", len(chunks))

    extractor: BaseExtractor = JsonExtractor(input_path=JSON_KW_PATH)
    # To switch extractors, replace the line above with e.g.:
    # extractor = CompositeExtractor([YakeExtractor(top_n=cfg.top_n), TfidfExtractor(top_n=cfg.top_n)])

    linker = CooccurrenceLinker(min_cooccurrence=cfg.min_cooccurrence)
    persister = NetworkxJsonPersister()
    pipeline = Pipeline(
        extractor=extractor,
        linker=linker,
        persister=persister,
    )
    pipeline.run(chunks=chunks, output_dir=run_dir)

    _update_latest_symlink(runs_dir, run_dir)
    logger.info("Updated: %s -> %s", os.path.join(runs_dir, "latest"), run_dir)


if __name__ == "__main__":
    load_dotenv()  # Load environment variables from .env if present
    main()
