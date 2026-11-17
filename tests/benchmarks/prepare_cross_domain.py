"""Regenerate unedited fixture text through the app's PDF OCR, outside CI."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from python_engine.tokensmith_pdf_preparation import SYSTEM, prepare_pdf
from python_engine.tokensmith_preparation import completion_client


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="New directory, separate from frozen fixtures")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    model = {"engine": "ollama", "ollamaModelName": "gemma4:26b", "contextLength": 16384}
    complete = completion_client(model)
    fixtures = ROOT / "tests/fixtures/cross-domain"
    for provenance_path in sorted(fixtures.glob("*.provenance.json")):
        expected = json.loads(provenance_path.read_text())
        name = provenance_path.name.removesuffix(".provenance.json")
        excerpt = fixtures / f"{name}.pdf"
        if hashlib.sha256(excerpt.read_bytes()).hexdigest() != expected["excerptSha256"]:
            raise ValueError(f"PDF excerpt hash mismatch: {excerpt}")
        pages = expected["physicalPages"]
        started = time.perf_counter()
        chunks, _ = prepare_pdf(excerpt, complete, args.output / "page-cache" / name,
                               progress=lambda done, total, message: print(message, flush=True))
        for index, chunk in enumerate(chunks, 1):
            chunk["tokensmithChunkId"] = f"{name}.{index:03}"
            chunk["originalPageStart"] = pages[chunk["pageStart"] - 1]
            chunk["originalPageEnd"] = pages[chunk["pageEnd"] - 1]
        provenance = {
            "excerptSha256": expected["excerptSha256"], "physicalPages": pages,
            "preparationModel": model, "promptSha256": hashlib.sha256(SYSTEM.encode()).hexdigest(),
            "seconds": time.perf_counter() - started,
        }
        (args.output / f"{name}.json").write_text(json.dumps(chunks, ensure_ascii=False, indent=2) + "\n")
        (args.output / provenance_path.name).write_text(json.dumps(provenance, indent=2) + "\n")
        print(f"{name}: {len(chunks)} chunks, {provenance['seconds']:.1f}s. Do not manually correct OCR output.")


if __name__ == "__main__":
    main()
