<p align="center">
  <img src="src/renderer/src/assets/tokensmith-mark.png" alt="TokenSmith icon" width="96" />
</p>

<h1 align="center">
  <img src="src/renderer/src/assets/tokensmith-logo.png" alt="TokenSmith" width="260" />
</h1>

TokenSmith is a desktop study app for asking questions about your course PDFs. It runs locally, retrieves relevant passages from your files, and shows sources with each answer.

<p align="center">
  <img src="docs/assets/tokensmith.png" alt="TokenSmith desktop app showing a source-backed chat answer" />
</p>

## Student Workflow

1. Install and start Ollama.
2. Download the recommended local models in TokenSmith.
3. Add a folder of course PDFs.
4. Ask questions in Chat.
5. Use source cards to check where an answer came from.
6. Continue with suggested follow-up questions when you want to study deeper.

## What TokenSmith Does

- Prepares PDFs for local search.
- Retrieves relevant passages before answering.
- Answers with source cards for checking the material.
- Suggests follow-up questions to keep a study session moving.
- Supports local chat and embedding models through Ollama.

## Install

Download the latest app from the GitHub Releases page:

https://github.com/georgia-tech-db/TokenSmith/releases

On first launch, TokenSmith will guide you through installing Ollama, downloading models, and adding PDFs.

## Notes

- TokenSmith is meant for your own course materials and PDFs you are allowed to use.
- Unsupported files are not prepared for search.

## Developer Setup

Install dependencies:

```sh
npm install
```

Build the local Python runtime used by development and packaging:

```sh
npm run setup:python-runtime
```

Start the app locally:

```sh
npm run dev
```

Run checks:

```sh
npm run typecheck
npm test
```

## Packaging

Create a macOS DMG:

```sh
npm run package:mac
```

Create a Windows portable zip:

```sh
npm run package:win
```
