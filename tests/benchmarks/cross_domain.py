"""Frozen, unedited app OCR excerpts; no OCR or generator calls in CI."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests/fixtures/cross-domain"
DOCUMENTS = [
    ("2", "Algorithms", "algorithms-reductions"),
    ("2", "Algorithms", "algorithms-subset-sum"),
    ("3", "LLM Compiler", "compiler"),
    ("4", "NIST storage security", "nist-recovery"),
    ("4", "NIST storage security", "nist-encryption"),
]


def fixture_paths():
    return [FIXTURES / f"{name}.json" for _, _, name in DOCUMENTS]


def load_chunks():
    chunks = []
    for collection, title, name in DOCUMENTS:
        for chunk in json.loads((FIXTURES / f"{name}.json").read_text()):
            chunks.append({**chunk, "_collectionId": collection,
                           "_documentTitle": title, "_fixture": f"{name}.pdf"})
    return chunks


def load_cases():
    return json.loads((ROOT / "tests/benchmarks/cross_domain_cases.json").read_text())
