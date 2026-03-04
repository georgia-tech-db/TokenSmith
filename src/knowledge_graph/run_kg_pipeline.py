import os
import pickle

from src.knowledge_graph.models import Chunk
from src.knowledge_graph.extractors import (
    BaseExtractor,
    CompositeExtractor,
    YakeExtractor,
    TfidfExtractor,
    KeyBERTExtractor,
    TextRankExtractor,
)
from src.knowledge_graph.linkers import CooccurrenceLinker
from src.knowledge_graph.persisters import NetworkxJsonPersister

from src.knowledge_graph.pipeline import Pipeline

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

# Will read chunks from textbook index chunks and metadata
CHUNKS_PKL = os.path.join(
    PROJECT_ROOT, "index", "sections", "textbook_index_chunks.pkl"
)
META_PKL = os.path.join(PROJECT_ROOT, "index", "sections", "textbook_index_meta.pkl")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "knowledge_graph")

MIN_COOCCURRENCE = 2  # prune edges that appear in fewer than X chunks
TOP_N = 10  # keywords per chunk


def load_chunks(chunks_path: str, meta_path: str) -> list[Chunk]:
    """Load pre-chunked text and metadata from pickle files into Chunk objects."""
    with open(chunks_path, "rb") as f:
        texts: list[str] = pickle.load(f)

    with open(meta_path, "rb") as f:
        metas: list[dict] = pickle.load(f)

    if len(texts) != len(metas):
        raise ValueError(
            f"Mismatch: {len(texts)} chunks vs {len(metas)} metadata entries"
        )

    chunks = []
    for i, (text, meta) in enumerate(zip(texts, metas)):
        chunk_id = meta.get("chunk_id", i)
        section_path = meta.get("section_path")
        if section_path is None:
            raise ValueError(f"Missing section_path in metadata for chunk {chunk_id}")
        if section_path.startswith("Chapter 12"):
            chunks.append(Chunk(id=chunk_id, text=text, metadata=meta))
    print("Only using chapter 12, debug mode")
    return chunks


def main() -> None:
    print(f"Loading chunks from:\n  {CHUNKS_PKL}\n  {META_PKL}")
    chunks = load_chunks(CHUNKS_PKL, META_PKL)
    print(f"Loaded {len(chunks)} chunks\n")
    extractor: BaseExtractor = CompositeExtractor(
        extractors=[
            YakeExtractor(top_n=TOP_N),
            TfidfExtractor(top_n=TOP_N),
            KeyBERTExtractor(top_n=TOP_N),
            TextRankExtractor(top_n=TOP_N),
        ]
    )
    linker = CooccurrenceLinker(min_cooccurrence=MIN_COOCCURRENCE)
    persister = NetworkxJsonPersister()
    pipeline = Pipeline(
        extractor=extractor, linker=linker, persister=persister, verbose=True
    )
    pipeline.run(chunks=chunks, output_dir=OUTPUT_DIR)


if __name__ == "__main__":
    main()
