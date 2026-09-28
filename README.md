<p align="center">
  <img src="src/renderer/src/assets/tokensmith-mark.png" alt="TokenSmith icon" width="120" />
</p>

# TokenSmith

**TokenSmith helps students learn from their course materials using local language models running on their laptop.**

- Add your textbooks, lecture notes, and papers.
- Work through difficult concepts with explanations and examples.
- Explore suggested questions.
- Test your understanding with short quizzes and feedback.

Local models run through Ollama. You can also connect a cloud model with an API key.

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

1. Install and start [Ollama](https://ollama.com/download), then use TokenSmith's setup guide to download a local embedding model for searching your documents.
2. Choose a local chat model, or select **Connect a cloud chat model** and use your own API key.
3. Add your course materials in **Library**. PDF, Markdown, and text files are supported.
4. Start with your own question or a suggested one. Open the source passages, follow up on an explanation, or choose **Quiz me** to practice.

## Project Health

[![CI](https://github.com/georgia-tech-db/TokenSmith/actions/workflows/ci.yml/badge.svg?branch=main&event=push)](https://github.com/georgia-tech-db/TokenSmith/actions/workflows/ci.yml)
[![Accuracy benchmark (retrieval)](https://github.com/georgia-tech-db/TokenSmith/actions/workflows/accuracy.yml/badge.svg?branch=main&event=push)](https://github.com/georgia-tech-db/TokenSmith/actions/workflows/accuracy.yml)
[![Frontend TypeScript coverage](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fgeorgia-tech-db%2FTokenSmith%2Fcodex%2Fcoverage-badge%2Ftypescript-coverage.json)](https://github.com/georgia-tech-db/TokenSmith/blob/codex/coverage-badge/coverage.md)
[![Backend Python coverage](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fgeorgia-tech-db%2FTokenSmith%2Fcodex%2Fcoverage-badge%2Fpython-coverage.json)](https://github.com/georgia-tech-db/TokenSmith/blob/codex/coverage-badge/coverage.md)

The benchmark checks whether search retrieves the expected evidence.

See [testing and benchmarks](docs/testing.md) for what the checks measure and how to run them.

## Contributing

[Report a problem](https://github.com/georgia-tech-db/TokenSmith/issues) or see the [development guide](docs/development.md) to run, build, and package the app.
