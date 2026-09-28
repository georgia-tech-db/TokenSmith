import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tests.benchmarks.test_buzzdb_embeddings import (
    all_chunks, cache_metadata, compute_embedding_bundle, sha256_bytes, write_embedding_bundle,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", nargs="?", type=Path, default=ROOT / "tmp/buzzdb-nomic-cache.npz")
    parser.add_argument("--cache-key", action="store_true")
    args = parser.parse_args()
    if args.cache_key:
        print(sha256_bytes(json.dumps(cache_metadata(all_chunks()), sort_keys=True).encode()))
        raise SystemExit(0)
    output = args.output
    bundle = compute_embedding_bundle()
    write_embedding_bundle(output, bundle)
    print(f"Wrote {len(bundle['chunk_ids'])} Nomic book embeddings to {output}; {bundle['metadata']['buildSeconds']:.2f}s")
