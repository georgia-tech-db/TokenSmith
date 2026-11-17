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

1. Install and start [Ollama](https://ollama.com/download), then use TokenSmith's setup guide to download a local embedding model for searching your documents.
2. Choose a local chat model, or select **Connect a cloud chat model** and use your own API key.
3. Add your course materials in **Library**. PDF, Markdown, and text files are supported.
4. Start with your own question or a suggested one. Open the source passages, follow up on an explanation, or switch to **Practice** and choose the documents you want to review.

GPU acceleration for document indexing and search is on by default. Ollama uses a compatible GPU when available and otherwise runs on the CPU. Turn off **GPU Acceleration for Document Search** in Settings to use only the CPU for these tasks. This setting does not change chat models.

## Chat attachments

Use **Attach file** in Chat to add a PDF, UTF-8 text or Markdown file, or PNG/JPEG
screenshot. The file is read in the background and its extracted text is saved
with the conversation. Remove the attachment before choosing another file.
Editing an earlier question keeps the file originally used with that question.

TokenSmith ranks matching attachment passages locally and combines them with
relevant passages from the selected Library documents. Expand **Attachment
evidence** under an answer to see which text and PDF pages were supplied to the
model. Attachment text stays outside the Library index. With a cloud chat model
selected, the selected passages are sent to that provider.

Attachments are limited to one file, 10 MiB, 20 PDF pages, and 120,000 extracted
characters. Images and rendered PDF pages are limited to 25 megapixels.
PDFium reads native text first. Screenshots and PDF pages with missing or
substantially corrupted text use RapidOCR locally on the CPU. OCR can misread
symbols and figures. Run `npm run setup:python-runtime` to add the OCR dependencies
and bundled models to an existing development runtime.

## Project Health

[![CI](https://github.com/georgia-tech-db/TokenSmith/actions/workflows/ci.yml/badge.svg?branch=main&event=push)](https://github.com/georgia-tech-db/TokenSmith/actions/workflows/ci.yml)
[![Accuracy benchmark (retrieval)](https://github.com/georgia-tech-db/TokenSmith/actions/workflows/accuracy.yml/badge.svg?branch=main&event=push)](https://github.com/georgia-tech-db/TokenSmith/actions/workflows/accuracy.yml)
[![Frontend TypeScript coverage](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fgeorgia-tech-db%2FTokenSmith%2Fcodex%2Fcoverage-badge%2Ftypescript-coverage.json)](https://github.com/georgia-tech-db/TokenSmith/blob/codex/coverage-badge/coverage.md)
[![Backend Python coverage](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fgeorgia-tech-db%2FTokenSmith%2Fcodex%2Fcoverage-badge%2Fpython-coverage.json)](https://github.com/georgia-tech-db/TokenSmith/blob/codex/coverage-badge/coverage.md)

The benchmark checks whether search retrieves the expected evidence.

See [testing and benchmarks](docs/testing.md) for what the checks measure and how to run them.

## Contributing

[Report a problem](https://github.com/georgia-tech-db/TokenSmith/issues) or see the [development guide](docs/development.md) to run, build, and package the app.
