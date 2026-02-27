import time
import argparse
import random
from src.knowledge_graph.models import Chunk
from src.knowledge_graph.run_kg_pipeline import load_chunks, CHUNKS_PKL, META_PKL, TOP_N
from src.knowledge_graph.extractors import (
    YakeExtractor,
    TfidfExtractor,
    KeyBERTExtractor,
    TextRankExtractor,
    SLMExtractor,
)
from src.knowledge_graph.extractors import BaseExtractor


def benchmark_extractor(name: str, extractor: BaseExtractor, chunks: list[Chunk]):
    print(f"\n{'=' * 50}")
    print(f"Benchmarking {name} on {len(chunks)} chunks...")
    print(f"{'=' * 50}")

    start_time = time.time()
    results = extractor.extract(chunks)
    elapsed = time.time() - start_time

    print(f"Total time: {elapsed:.3f} s")
    print(f"Time per chunk: {elapsed / len(chunks):.3f} s")

    # Display the first 3 results to inspect extraction quality
    print("\nSample extractions:")
    for i, res in enumerate(results[:3]):
        print(f"  Chunk {res.chunk_id}:")
        print(f"    Text snippet: {chunks[i].text[:100]}...")
        print(f"    Keywords: {res.nodes}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark different keyword extractors."
    )
    parser.add_argument(
        "--num_chunks", type=int, default=10, help="Number of chunks to benchmark on."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    args = parser.parse_args()

    print(f"Loading chunks from:\n  {CHUNKS_PKL}\n  {META_PKL}")
    all_chunks = load_chunks(CHUNKS_PKL, META_PKL)
    print(f"Loaded {len(all_chunks)} total chunks.")

    # Select sample chunks for benchmark
    random.seed(args.seed)
    sample_chunks = random.sample(all_chunks, args.num_chunks)
    print(f"Selected {len(sample_chunks)} chunks for benchmarking.\n")

    extractors = [
        ("YAKE", YakeExtractor(top_n=TOP_N)),
        ("TF-IDF", TfidfExtractor(top_n=TOP_N)),
        ("KeyBERT", KeyBERTExtractor(top_n=TOP_N)),
        ("TextRank", TextRankExtractor(top_n=TOP_N)),
        ("SLM", SLMExtractor()),
    ]

    for name, extractor in extractors:
        benchmark_extractor(name, extractor, sample_chunks)


if __name__ == "__main__":
    main()
