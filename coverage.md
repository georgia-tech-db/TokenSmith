## Unit-Test Line Coverage

| Scope | Covered / total lines | Coverage |
| --- | ---: | ---: |
| Frontend / Electron (TypeScript / TSX) | 2954/13803 | 21.4% |
| Backend (Python) | 2549/2908 | 87.7% |

Includes untested files in `src/` and `python_engine/`; excludes TypeScript declaration files.
TypeScript coverage includes the UI, Electron main/preload, and shared modules, not only the renderer.
Measured with c8 (V8 line coverage) and coverage.py (executable Python lines).
This measures unit-test execution, not answer accuracy or full end-to-end coverage.

<details><summary>Per-file coverage</summary>

| File | Covered / total lines | Coverage |
| --- | ---: | ---: |
| `python_engine/tokensmith_cleaning.py` | 145/159 | 91.2% |
| `python_engine/tokensmith_engine.py` | 1317/1536 | 85.7% |
| `python_engine/tokensmith_preparation_job.py` | 103/118 | 87.3% |
| `python_engine/tokensmith_preparation.py` | 148/166 | 89.2% |
| `python_engine/tokensmith_store.py` | 836/929 | 90.0% |
| `src/main/engine/cloud-generator-service.ts` | 211/211 | 100.0% |
| `src/main/engine/engine-service.ts` | 0/171 | 0.0% |
| `src/main/engine/ollama-library-search.ts` | 211/223 | 94.6% |
| `src/main/engine/ollama-service.ts` | 384/1008 | 38.1% |
| `src/main/engine/question-rewrite.ts` | 55/59 | 93.2% |
| `src/main/engine/remote-chat-parameters.ts` | 9/9 | 100.0% |
| `src/main/engine/remote-chat-service.ts` | 297/349 | 85.1% |
| `src/main/engine/remote-generator-network.ts` | 6/7 | 85.7% |
| `src/main/engine/remote-model-secrets.ts` | 62/67 | 92.5% |
| `src/main/engine/study-chat-format.ts` | 757/783 | 96.7% |
| `src/main/engine/study-engine-core.ts` | 83/90 | 92.2% |
| `src/main/index.ts` | 0/550 | 0.0% |
| `src/main/models/local-model-service.ts` | 0/35 | 0.0% |
| `src/main/python/python-engine-service.ts` | 196/803 | 24.4% |
| `src/preload/index.ts` | 0/137 | 0.0% |
| `src/renderer/src/App.tsx` | 0/7220 | 0.0% |
| `src/renderer/src/chat-interactions.ts` | 15/15 | 100.0% |
| `src/renderer/src/ChatModelPicker.tsx` | 0/67 | 0.0% |
| `src/renderer/src/CloudGeneratorDialog.tsx` | 0/176 | 0.0% |
| `src/renderer/src/ConversationViewport.tsx` | 0/216 | 0.0% |
| `src/renderer/src/LibraryWorkspace.tsx` | 0/200 | 0.0% |
| `src/renderer/src/main.tsx` | 0/12 | 0.0% |
| `src/renderer/src/markdown-source.ts` | 87/87 | 100.0% |
| `src/renderer/src/MarkdownSourceViewer.tsx` | 75/105 | 71.4% |
| `src/renderer/src/MessageText.tsx` | 34/35 | 97.1% |
| `src/renderer/src/QuestionEditor.tsx` | 0/54 | 0.0% |
| `src/renderer/src/ThemePicker.tsx` | 0/50 | 0.0% |
| `src/shared/app-state.ts` | 0/219 | 0.0% |
| `src/shared/bridge.ts` | 0/89 | 0.0% |
| `src/shared/cleaning.ts` | 0/122 | 0.0% |
| `src/shared/cloud-generators.ts` | 51/52 | 98.1% |
| `src/shared/engine.ts` | 0/159 | 0.0% |
| `src/shared/model-defaults.ts` | 87/87 | 100.0% |
| `src/shared/model-providers.ts` | 52/52 | 100.0% |
| `src/shared/ollama.ts` | 66/66 | 100.0% |
| `src/shared/preparation.ts` | 50/50 | 100.0% |
| `src/shared/quiz.ts` | 80/82 | 97.6% |
| `src/shared/retrieval-budget.ts` | 21/21 | 100.0% |
| `src/shared/study-chat-pipeline.ts` | 65/65 | 100.0% |

</details>

Measured commit: 176a3a11131c50421bf81e6107a63830355c0cd9

[CI run](https://github.com/georgia-tech-db/TokenSmith/actions/runs/35757122420)
