<p align="center">
  <img src="src/renderer/src/assets/tokensmith-mark.png" alt="TokenSmith icon" width="120" />
</p>

# TokenSmith

**TokenSmith helps students learn from their course materials using local language models running on their laptop.**

- **Practice Mode:** Test your understanding with quizzes. Reinforce what topics you’ve mastered and identify topics to revisit.
- **Chat Mode:** Add textbooks, lecture notes, and papers. Explore concepts through explanations, examples, and suggested questions.

Local models run through Ollama. You can also connect a cloud model with an API key.

**[Download](#download)** | **[Watch the demo](#demo)** | **[Get started](#get-started)**

## Download

[Latest release](https://github.com/georgia-tech-db/TokenSmith/releases/latest) | [Release notes for v0.1.16](https://github.com/georgia-tech-db/TokenSmith/releases/tag/v0.1.16)

| Platform | Download v0.1.16 |
| --- | --- |
| macOS (Apple Silicon) | [DMG](https://github.com/georgia-tech-db/TokenSmith/releases/download/v0.1.16/TokenSmith-0.1.16-mac-arm64.dmg) |
| macOS (Intel, macOS 15+) | [DMG](https://github.com/georgia-tech-db/TokenSmith/releases/download/v0.1.16/TokenSmith-0.1.16-mac-x64.dmg) |
| Windows (x64) | [Portable ZIP](https://github.com/georgia-tech-db/TokenSmith/releases/download/v0.1.16/TokenSmith-0.1.16-win-x64.zip) |
| Linux (x64) | [AppImage](https://github.com/georgia-tech-db/TokenSmith/releases/download/v0.1.16/TokenSmith-0.1.16-linux-x64.AppImage) or [Debian package](https://github.com/georgia-tech-db/TokenSmith/releases/download/v0.1.16/tokensmith_0.1.16_amd64.deb) |

## Demo

https://github.com/user-attachments/assets/933908db-2117-4e90-9628-74e2038bbbfa

## Get Started

1. Install [Ollama](https://ollama.com/download), then follow TokenSmith’s setup guide to download a local embedding model for searching your documents.
2. Download a local chat model, or select **Connect a cloud chat model** to use your own API key.
3. Add your course documents to **Library**. TokenSmith supports Markdown, PDF, and text files.
4. **Choose your learning mode:**
   - **Chat Mode:** Ask your own question or explore a suggested one, follow up on explanations, and open source passages.
   - **Practice Mode:** Choose a library, test your understanding with quizzes, and identify concepts to revisit.

## Project Health

[![CI](https://github.com/georgia-tech-db/TokenSmith/actions/workflows/ci.yml/badge.svg?branch=main&event=push)](https://github.com/georgia-tech-db/TokenSmith/actions/workflows/ci.yml)
[![Accuracy benchmark (retrieval)](https://github.com/georgia-tech-db/TokenSmith/actions/workflows/accuracy.yml/badge.svg?branch=main&event=push)](https://github.com/georgia-tech-db/TokenSmith/actions/workflows/accuracy.yml)
[![Frontend TypeScript coverage](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fgeorgia-tech-db%2FTokenSmith%2Fcodex%2Fcoverage-badge%2Ftypescript-coverage.json)](https://github.com/georgia-tech-db/TokenSmith/blob/codex/coverage-badge/coverage.md)
[![Backend Python coverage](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fgeorgia-tech-db%2FTokenSmith%2Fcodex%2Fcoverage-badge%2Fpython-coverage.json)](https://github.com/georgia-tech-db/TokenSmith/blob/codex/coverage-badge/coverage.md)

The benchmark checks whether search retrieves the expected evidence.

See [testing and benchmarks](docs/testing.md) for what the checks measure and how to run them.

## Contributing

[Report a problem](https://github.com/georgia-tech-db/TokenSmith/issues) or see the [development guide](docs/development.md) to run, build, and package the app.
