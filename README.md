# TokenSmith Electron

Student-focused desktop app for local, source-backed study chat with TokenSmith's RAG study pipeline.

## What Works Now

- Electron desktop shell with Chat, Library, Models, and Settings.
- Local material import for PDF, Markdown, text files, and folders.
- Python-backed extraction/indexing for readable PDFs through pypdfium2, plus Markdown, text files, and folders.
- TokenSmith-style chunking with word counts, page estimates for PDFs, and local app-data storage.
- Local FAISS vector retrieval over embedded chunks before chat responses.
- Source cards under answers when matching chunks are found.
- Python Study Engine routing for extraction, retrieval, and source-backed chat.
- Optional GGUF inference through `llama-cpp-python` when it is installed in the local Python environment.

## Scripts

- `npm run dev` starts the Electron app with the Vite renderer.
- `npm run typecheck` checks the Electron, preload, shared, and React code.
- `npm run setup:python-runtime` builds the local `app_runtime/python` bundle used by `npm run dev` and packaging.
- `npm test` runs TypeScript unit tests, Python unit tests, and the toy-PDF integration test.
- `npm run coverage` runs c8 for TypeScript unit coverage and coverage.py for the Python engine.
- `npm run build` creates production bundles.
- `npm run package:mac` creates a versioned student DMG at `release/TokenSmith-<version>-mac-<arch>.dmg`.
- `npm run package:win` creates a versioned portable Windows zip at `release/TokenSmith-<version>-win-<arch>.zip`.
- `npm run preview` runs the built app locally.

## Student DMG

Run this on macOS:

```sh
npm run package:mac
```

The script builds the app, creates `release/mac/TokenSmith.app`, ad-hoc signs it when possible, and writes a versioned DMG named from `package.json`, for example `TokenSmith-0.1.0-mac-arm64.dmg`.

The DMG includes the Electron app, Python worker, and bundled `app_runtime/python` runtime produced during packaging.

## Student Windows Package

Run this on Windows:

```sh
npm run package:win
```

The script builds the app, creates `release/win/TokenSmith`, and writes a portable zip named from `package.json`.

The GitHub Actions workflow `.github/workflows/windows-build.yml` can produce the same Windows zip from `windows-latest` and upload it as a workflow artifact. It runs manually through `workflow_dispatch` and on version tags like `v0.1.3`.

## Student Flow

1. Open Library and add a readable course file or folder.
2. TokenSmith extracts text and creates searchable study chunks.
3. Open Chat and ask a course question.
4. Chat retrieves matching chunks, answers in a tutor style, and shows the source cards.

## Python Engine

Electron starts `python_engine/tokensmith_engine.py` as a local worker and speaks newline-delimited JSON over stdin/stdout. The worker supports:

- `health`
- `index_material`
- `search`
- `chat`

For local development, build the same `app_runtime/python` layout used by GitHub Actions:

```sh
npm run setup:python-runtime
```

The setup script follows the GitHub Actions shape: it runs `python -m pip install -r requirements-runtime.txt`, computes that Python root, and copies it into `app_runtime/python`. Local dev, tests, and packaging then use `app_runtime/python` directly.
