import time
import argparse
import random

from dotenv import load_dotenv
import os
from src.knowledge_graph.models import Chunk
from src.knowledge_graph.build import load_chunks, CHUNKS_PKL, META_PKL
from src.knowledge_graph.extractors import (
    YakeExtractor,
    TfidfExtractor,
    KeyBERTExtractor,
    TextRankExtractor,
    SLMExtractor,
    OpenRouterExtractor,
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
        print(f"    Keywords: {res.keywords}")


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
    top_n = 20

    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY environment variable not set. Please set it to benchmark the OpenRouter extractor.")

    extractors = [
        ("YAKE", YakeExtractor(top_n=top_n)),
        ("TF-IDF", TfidfExtractor(top_n=top_n)),
        ("KeyBERT", KeyBERTExtractor(top_n=top_n)),
        ("TextRank", TextRankExtractor(top_n=top_n)),
        ("SLM", SLMExtractor(top_n=top_n)),
        ("OpenRouter", OpenRouterExtractor(api_key=api_key,
         model='qwen/qwen3-next-80b-a3b-instruct', top_n=top_n)),
    ]

    for name, extractor in extractors:
        benchmark_extractor(name, extractor, sample_chunks)


if __name__ == "__main__":
    load_dotenv()
    main()
