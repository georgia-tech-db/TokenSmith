"""Runs the engine's NLI answer check over the BuzzDB grounding cases.

Prints one JSON document: every case answer with the claims the engine
extracted, their per-passage entailment and contradiction, and how long the
check took. test_buzzdb_nli.mjs turns that into app statuses and metrics.
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from python_engine import tokensmith_engine as engine
from python_engine import tokensmith_nli as nli

FIXTURE_PATH = ROOT / "tests" / "fixtures" / "buzzdb" / "buzzdb-book.tokensmith.md"
GROUNDING_CASES_PATH = ROOT / "tests" / "benchmarks" / "buzzdb_grounding_cases.json"
NLI_CASES_PATH = ROOT / "tests" / "benchmarks" / "buzzdb_nli_cases.json"


def main() -> None:
    model_dir = sys.argv[1] if len(sys.argv) > 1 else None
    chunks = {
        chunk.get("tokensmithChunkId"): chunk
        for chunk in engine.chunk_tokensmith_markdown(FIXTURE_PATH.read_text(encoding="utf-8"))
    }
    grounding = {case["id"]: case for case in json.loads(GROUNDING_CASES_PATH.read_text(encoding="utf-8"))}

    started = time.perf_counter()
    nli.load_model(model_dir)
    load_ms = (time.perf_counter() - started) * 1000

    results = []
    for group in json.loads(NLI_CASES_PATH.read_text(encoding="utf-8")):
        # Older cases borrow their question and passages from the grounding
        # benchmark; newer ones carry their own.
        case = grounding.get(group["caseId"]) or {
            "question": group["question"],
            "expectedChunkIds": group["chunkIds"],
            "forbiddenChunkIds": group.get("distractorChunkIds", []),
        }
        # The passages retrieval should return, plus the known distractors, so
        # every answer is checked against on-topic and off-topic text alike.
        chunk_ids = case["expectedChunkIds"] + case.get("forbiddenChunkIds", [])
        passages = [chunks[chunk_id]["text"] for chunk_id in chunk_ids]
        for answer in group["answers"]:
            question = answer.get("question") or case["question"]
            started = time.perf_counter()
            verification = nli.verify_claims(answer["text"], passages, model_dir, question)
            results.append(
                {
                    "caseId": group["caseId"],
                    "split": group.get("split", "dev"),
                    "kind": answer["kind"],
                    "expect": answer["expect"],
                    "question": question,
                    "passageCount": len(passages),
                    "durationMs": (time.perf_counter() - started) * 1000,
                    "verification": verification,
                }
            )

    json.dump({"modelLoadMs": load_ms, "results": results}, sys.stdout)


if __name__ == "__main__":
    main()
