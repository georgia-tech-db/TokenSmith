import argparse
import logging
import os

from dotenv import load_dotenv

from src.knowledge_graph.build import (
    CHUNKS_PKL,
    JSON_KW_PATH,
    META_PKL,
    OUTPUT_DIR,
    PROJECT_ROOT,
    load_chunks,
    create_run_dir,
    setup_input_dir,
    write_config,
    update_latest_symlink,

)
from src.knowledge_graph.canonicalizer import Canonicalizer
from src.knowledge_graph.extractors import BaseExtractor, JsonExtractor
from src.knowledge_graph.linkers import CooccurrenceLinker
from src.knowledge_graph.persisters import NetworkxJsonPersister
from src.knowledge_graph.pipeline import Pipeline
from src.knowledge_graph.io import load_run_chunks
from src.knowledge_graph.openrouter_client import OpenRouterClient
from src.knowledge_graph.section_tree import build_section_tree, save_section_tree
from src.knowledge_graph.summary_tree import build_summary_index
from src.knowledge_graph.models import KGPipelineConfig

logger = logging.getLogger(__name__)


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
    parser.add_argument(
        "--summarize",
        action="store_true",
        help="Build LLM summary index after the section tree (requires OPENROUTER_API_KEY).",
    )
    parser.add_argument(
        "--summary-model",
        default="openai/gpt-4o-mini",
        help="OpenRouter model used for chunk/section summarization.",
    )
    parser.add_argument(
        "--embed-model",
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer model for embedding summaries.",
    )
    parser.add_argument(
        "--chunk-window",
        type=int,
        default=3,
        help="Number of adjacent chunks summarized together at the leaf level.",
    )
    args = parser.parse_args()

    cfg = KGPipelineConfig.from_yaml(args.config)
    logger.info("Loaded config from %s", args.config)

    runs_dir = os.path.join(OUTPUT_DIR, "runs")
    run_dir = create_run_dir(runs_dir)
    logger.info("Run directory: %s", run_dir)

    setup_input_dir(run_dir)
    write_config(run_dir, cfg)

    logger.info("Loading chunks from:\n  %s\n  %s", CHUNKS_PKL, META_PKL)
    chunks = load_chunks(CHUNKS_PKL, META_PKL)
    logger.info("Loaded %d chunks", len(chunks))

    extractor: BaseExtractor = JsonExtractor(input_path=JSON_KW_PATH)
    # To switch extractors, replace the line above with e.g.:
    # extractor = CompositeExtractor([YakeExtractor(top_n=cfg.top_n), TfidfExtractor(top_n=cfg.top_n)])

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "OPENROUTER_API_KEY environment variable must be set for canonicalization."
        )

    c = cfg.canonicalization
    canonicalizer = Canonicalizer(
        embedding_model=c.embed_model,
        corpus_description=cfg.corpus_description,
        api_key=api_key,
        llm_model=c.llm_model,
        similarity_threshold=c.similarity_threshold,
        max_group_size=c.max_group_size,
        batch_size=c.batch_size,
    )
    # canonicalizer = MockCanonicalizer("debug/canonicalization_cache.json")
    linker = CooccurrenceLinker(min_cooccurrence=cfg.min_cooccurrence)
    persister = NetworkxJsonPersister()
    pipeline = Pipeline(
        extractor=extractor,
        linker=linker,
        persister=persister,
        canonicalizer=canonicalizer,
    )
    graph = pipeline.run(chunks=chunks, output_dir=run_dir)

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
    client = OpenRouterClient(api_key, retries=2)
    summarize_fn = lambda messages: client.chat(st.summary_model, messages)
    build_summary_index(
        section_tree=tree,
        chunks=chunk_texts,
        summarize_fn=summarize_fn,
        embed_model=st.embed_model,
        chunk_window=st.chunk_window,
        run_dir=run_dir,
    )
    logger.info("Summary index saved to %s", run_dir)

    update_latest_symlink(runs_dir, run_dir)
    logger.info("Updated: %s -> %s", os.path.join(runs_dir, "latest"), run_dir)


if __name__ == "__main__":
    load_dotenv()
    main()
