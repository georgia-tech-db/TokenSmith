import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "tests" / "benchmarks" / "buzzdb-fastembed-cache.npz"
sys.path.insert(0, str(ROOT))

from tests.benchmarks.test_buzzdb_fastembed import compute_embedding_bundle, write_embedding_bundle


def main():
    output_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUTPUT
    bundle = compute_embedding_bundle()
    write_embedding_bundle(output_path, bundle)
    measurements = bundle["metadata"]["measurements"]
    print(
        "Wrote FastEmbed BuzzDB cache: "
        f"{output_path}; chunks={measurements['chunk_count']}; "
        f"load={measurements['model_load_seconds']:.2f}s; "
        f"warmup={measurements['warmup_seconds']:.2f}s; "
        f"embed={measurements['passage_embedding_seconds']:.2f}s; "
        f"query_embed={measurements['query_embedding_seconds']:.2f}s"
    )


if __name__ == "__main__":
    main()
