"""Isolated real-hybrid index for the opt-in answer benchmark; never uses app data."""
from contextlib import redirect_stdout
import json
import sys

from tests.benchmarks.test_buzzdb_embeddings import BuzzDBEmbeddingBenchmarkTests
from tests.benchmarks.test_buzzdb_grounding import normalize_text, validate_grounding_case


def main():
    benchmark = BuzzDBEmbeddingBenchmarkTests
    try:
        # Cache-building progress must not be mistaken for a JSON RPC response.
        with redirect_stdout(sys.stderr):
            benchmark.setUpClass()
        instance = benchmark()
        print(json.dumps({"ready": True, "cache": benchmark.bundle["metadata"]}), flush=True)
        for line in sys.stdin:
            request = json.loads(line)
            try:
                if request["command"] == "search":
                    result = instance.retrieve_sources(request["query"], request["limit"])
                elif request["command"] == "validate":
                    result = validate_grounding_case(request["case"], [], request["chunkIds"],
                                                     normalize_text(request["context"]))
                else:
                    raise ValueError("Unknown benchmark command")
                print(json.dumps({"id": request["id"], "ok": True, "result": result}), flush=True)
            except Exception as error:
                print(json.dumps({"id": request["id"], "ok": False, "error": str(error)}), flush=True)
    finally:
        benchmark.doClassCleanups()


if __name__ == "__main__":
    main()
