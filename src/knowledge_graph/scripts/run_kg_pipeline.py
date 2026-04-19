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
    load_chunks,
    META_PKL,
    CHUNKS_PKL,
)
from src.knowledge_graph.models import KGPipelineConfig
from src.knowledge_graph.pipeline import build_kg
from src.knowledge_graph.summary_tree import build_summary_index
from src.knowledge_graph.openrouter_client import OpenRouterClient
from src.knowledge_graph.io import load_run_chunks
from src.knowledge_graph.section_tree import build_section_tree, save_section_tree
from src.knowledge_graph.canonicalizer import Canonicalizer
from src.knowledge_graph.linkers import CooccurrenceLinker


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

    c = cfg.canonicalization
    canonicalizer = Canonicalizer(
        embedding_model=c.embed_model,
        corpus_description=cfg.corpus_description,
        api_key=args.api_key or os.environ.get("OPENROUTER_API_KEY", ""),
        llm_model=c.llm_model,
        similarity_threshold=c.similarity_threshold,
        max_group_size=c.max_group_size,
        batch_size=c.batch_size,
    )

    linker = CooccurrenceLinker(min_cooccurrence=cfg.min_cooccurrence)

    chunks = load_chunks(
        CHUNKS_PKL,
        META_PKL,
        chapter_filter=chapter_filter,
        exclude_chapters=exclude_chapters,
    )
    logger.info("Loaded %d chunks", len(chunks))
    graph = build_kg(
        output_dir=run_dir,
        chunks=chunks,
        extractor=extractor,
        linker=linker,
        canonicalizer=canonicalizer,
    )

    logger.info("Building section tree...")
    tree = build_section_tree(chunks, graph)
    tree_path = save_section_tree(tree, run_dir)
    level_counts: dict[int, int] = {}
    for node in tree.node_index.values():
        level_counts[node.level] = level_counts.get(node.level, 0) + 1
    level_labels = {1: "chapters", 2: "sections", 3: "subsections"}
    for level, count in sorted(level_counts.items()):
        label = level_labels.get(level, f"level-{level} nodes")
        logger.info("  %4d %s", count, label)
    logger.info("  Saved: %s", tree_path)

    st = cfg.summary_tree
    logger.info(
        "Building summary index (model=%s, chunk_window=%d)...",
        st.summary_model,
        st.chunk_window,
    )
    chunk_texts = load_run_chunks(os.path.join(run_dir, "chunks.json"))
    client = OpenRouterClient(args.api_key, retries=2)
    build_summary_index(
        client=client,
        summary_model=st.summary_model,
        section_tree=tree,
        chunks=chunk_texts,
        embed_model=st.embed_model,
        chunk_window=st.chunk_window,
        run_dir=run_dir,
    )
    logger.info("Summary index saved to %s", run_dir)

    update_latest_symlink(run_dir)
    logger.info("Updated: %s -> %s", os.path.join(RUNS_DIR, "latest"), run_dir)


if __name__ == "__main__":
    load_dotenv()
    main()
