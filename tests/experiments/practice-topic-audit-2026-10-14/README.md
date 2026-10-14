# BuzzDBBook Practice audit — 14 October 2026

`summary.json` records the twelve requests, generated statements, keys, timings,
and book locations. `review.json` records the separate qualitative review:
eight sound questions, three needing qualification, and one request blocked
after two invalid citation packages. This is one small, model-assisted review,
not a measured population accuracy rate or an independent human benchmark.

The opt-in runner uses the current local model/settings and the production
retrieval, Practice, and provider code against a temporary SQLite snapshot.
It requires a loaded library named BuzzDBBook, Ollama generator and embedding
models, and the project's Python runtime. It does not change the user's history.

```sh
TOKENSMITH_AUDIT_STATE="/path/to/TokenSmith/app-data" node tests/experiments/practice-topic-audit-2026-10-14/run.mjs
BUZZDB_BOOK="/path/to/buzzdb-book.tokensmith.md" node tests/experiments/practice-topic-audit-2026-10-14/verify.mjs
python3 tests/experiments/practice-topic-audit-2026-10-14/render.py
```

The environment variables default to the original test's macOS directory layout
under the current user's home. The runner writes `results.json`; rerunning it
replaces that local capture. Back it up first if needed. Review new generations
independently and update `review.json` and its capture hash before rendering;
the renderer refuses to apply this run's judgments to different results.

Full captures, the citation verification output, and the rendered report stay
local and are ignored by Git because they include full course passages and
machine-specific paths. No textbook or paper PDFs are bundled with this audit.
