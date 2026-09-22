<p align="center">
  <img src="src/renderer/src/assets/tokensmith-mark.png" alt="TokenSmith icon" width="120" />
</p>

# TokenSmith

**Study with your own textbooks, notes, and papers.**

Work through a difficult concept, ask for an example, or try a short quiz to see what you've understood. TokenSmith suggests questions to explore next, gives feedback on your quiz answers, and lets you return to the original passages as you study.

Highlight something you want explained, follow an idea further, or revisit an earlier discussion when you're reviewing.

Use a local model with Ollama or a cloud model with your own API key.

**[Download](#download)** | **[Watch the demo](#demo)** | **[Get started](#get-started)**

## Download

[Latest release](https://github.com/georgia-tech-db/TokenSmith/releases/latest) | [Release notes for v0.1.13](https://github.com/georgia-tech-db/TokenSmith/releases/tag/v0.1.13)

| Platform | Download v0.1.13 |
| --- | --- |
| macOS (Apple Silicon) | [DMG](https://github.com/georgia-tech-db/TokenSmith/releases/download/v0.1.13/TokenSmith-0.1.13-mac-arm64.dmg) |
| Windows (x64) | [Portable ZIP](https://github.com/georgia-tech-db/TokenSmith/releases/download/v0.1.13/TokenSmith-0.1.13-win-x64.zip) |
| Linux (x64) | [AppImage](https://github.com/georgia-tech-db/TokenSmith/releases/download/v0.1.13/TokenSmith-0.1.13-linux-x64.AppImage) or [Debian package](https://github.com/georgia-tech-db/TokenSmith/releases/download/v0.1.13/tokensmith_0.1.13_amd64.deb) |

## Demo

https://github.com/user-attachments/assets/933908db-2117-4e90-9628-74e2038bbbfa

## Get Started

1. Install and start [Ollama](https://ollama.com/download), then use TokenSmith's setup guide to download Nomic for searching your documents.
2. Choose a local chat model, or select **Connect a cloud model** and use your own API key. Cloud answers still use the local document-search setup above.
3. Add your course materials in **Library**. PDF, Markdown, and text files are supported; **Basic** is the default preparation mode.
4. Start with your own question or a suggested one. Open the source passages, follow up on an explanation, or choose **Quiz me** to practice.

When you use a cloud model, chat messages and relevant document excerpts are sent to that provider, and API charges may apply. Optional AI document preparation also sends document text to the selected model if it is a cloud model.

## Project Health

[![CI](https://github.com/georgia-tech-db/TokenSmith/actions/workflows/ci.yml/badge.svg?branch=main&event=push)](https://github.com/georgia-tech-db/TokenSmith/actions/workflows/ci.yml)
[![Accuracy benchmark (retrieval)](https://github.com/georgia-tech-db/TokenSmith/actions/workflows/accuracy.yml/badge.svg?branch=main&event=push)](https://github.com/georgia-tech-db/TokenSmith/actions/workflows/accuracy.yml)
[![Frontend TypeScript coverage](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fgeorgia-tech-db%2FTokenSmith%2Fcodex%2Fcoverage-badge%2Ftypescript-coverage.json)](https://github.com/georgia-tech-db/TokenSmith/blob/codex/coverage-badge/coverage.md)
[![Backend Python coverage](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fgeorgia-tech-db%2FTokenSmith%2Fcodex%2Fcoverage-badge%2Fpython-coverage.json)](https://github.com/georgia-tech-db/TokenSmith/blob/codex/coverage-badge/coverage.md)

The benchmark checks whether search retrieves the expected evidence, **not whether generated answers are correct**. Coverage badges report unit-test line coverage separately for TypeScript (including Electron and shared code) and Python.

See [testing and benchmarks](docs/testing.md) for what the checks measure and how to run them.

## Contributing

[Report a problem](https://github.com/georgia-tech-db/TokenSmith/issues) or see the [development guide](docs/development.md) to run, build, and package the app.
