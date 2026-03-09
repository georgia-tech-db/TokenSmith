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

    sample_chunks = [
        Chunk(
            id=0,
            # text="The hierarchical architecture combines the characteristics of shared-memory, shared-disk, and shared-nothing architectures. At the top level, the system consists of nodes that are connected by an interconnection network and do not share disks or memory with one another. Thus, the top level is a shared-nothing architecture. Each node of the system could actually be a shared-memory system with a few processors. Alternatively, each node could be a shared-disk system, and each of the systems sharing a set of disks could be a shared-memory system. Thus, a system could be built as a hierarchy, with shared-memory architecture with a few processors at the base, and a shared-nothing architecture at the top, with possibly a shared-disk architecture in the middle. Figure 20.5d illustrates a hierarchical architecture with shared-memory nodes connected together in a shared-nothing architecture. Parallel database systems today typically run on a hierarchical architecture, where each node supports shared-memory parallelism, with multiple nodes interconnected in a shared-nothing manner.",
            text=" . - Page 561 Optical storage . The digital video disk (DVD) is an optical storage medium, with data written and read back using a laser light source. The Blu-ray DVD format has a capacity of 27 gigabytes to 128 gigabytes, depending on the number of layers supported. Although the original (and still main) use of DVDs was to store video data, they are capable of storing any type of digital data, including backups of database contents. DVDs are not suitable for storing active database data since the time required to access a given piece of data can be quite long compared to the time taken by a magnetic disk. - Some DVD versions are read-only, written at the factory where they are produced, other versions support write-once, allowing them to be written once, but not overwritten, and some versions can be rewritten multiple times. Disks that can be written only once are - called write-once, read-many (WORM) disks. Optical disk jukebox systems contain a few drives and numerous disks that can be loaded into one of the drives automatically (by a robot arm) on demand. - Tape storage . T ape storage is used primarily for backup and archival data. Archival data refers to data that must be stored safely for a long period of time, often for legal reasons. Magnetic tape is cheaper than disks and can safely store data for many years. However, access to data is much slower because the tape must be accessed sequentially from the beginning of the tape; tapes can be very long, requiring tens to hundreds of seconds to access data. For this reason, tape storage is referred to as sequential-access storage. In contrast, magnetic disk and SSD storage are referred to as direct-access storage because it is possible to read data from any location on disk. Tapes have a high capacity (1 to 12 terabyte capacities are currently available), and can be removed from the tape drive   ",
            metadata={"source": "test"},
        )
    ]
    top_n = 20

    extractors = [
        ("YAKE", YakeExtractor(top_n=top_n)),
        ("TF-IDF", TfidfExtractor(top_n=top_n)),
        ("KeyBERT", KeyBERTExtractor(top_n=top_n)),
        ("TextRank", TextRankExtractor(top_n=top_n)),
        ("SLM", SLMExtractor(top_n=top_n)),
    ]

    for name, extractor in extractors:
        benchmark_extractor(name, extractor, sample_chunks)


if __name__ == "__main__":
    main()
