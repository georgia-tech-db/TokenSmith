"""Read the installed app's index through a temporary snapshot for live experiments."""
import json
from pathlib import Path
import sqlite3
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python_engine"))
import tokensmith_engine as engine


with tempfile.TemporaryDirectory(prefix="tokensmith-model-comparison-") as directory:
    original_path = Path(sys.argv[1]) / "tokensmith.sqlite"
    with sqlite3.connect(original_path.as_uri() + "?mode=ro", uri=True) as original:
        with sqlite3.connect(str(Path(directory) / "tokensmith.sqlite")) as snapshot:
            original.backup(snapshot)
    vector_search = engine.vector_search
    vector_counts = []

    def measured_vector_search(*args, **kwargs):
        hits = vector_search(*args, **kwargs)
        vector_counts.append(len(hits))
        return hits

    engine.vector_search = measured_vector_search
    for line in sys.stdin:
        try:
            payload = json.loads(line)
            vector_counts.clear()
            result = engine.search_library({**payload, "userDataPath": directory, "searchMode": "hybrid"})
            if not vector_counts or not any(vector_counts):
                raise RuntimeError("No vector hits: refusing to report a keyword-only run as hybrid.")
            print(json.dumps({"ok": True, "result": result, "vectorHits": sum(vector_counts)}), flush=True)
        except Exception as error:
            print(json.dumps({"ok": False, "error": str(error)}), flush=True)
