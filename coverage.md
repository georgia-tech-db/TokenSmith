## Unit-Test Line Coverage

| Scope | Covered / total lines | Coverage |
| --- | ---: | ---: |
| Frontend / Electron (TypeScript / TSX) | 4392/16203 | 27.1% |
| Backend (Python) | 2074/2312 | 89.7% |

Includes untested files in `src/` and `python_engine/`; excludes TypeScript declaration files.
TypeScript coverage includes the UI, Electron main/preload, and shared modules, not only the renderer.
Measured with c8 (V8 line coverage) and coverage.py (executable Python lines).
This measures unit-test execution, not answer accuracy or full end-to-end coverage.

<details><summary>Per-file coverage</summary>

| File | Covered / total lines | Coverage |
| --- | ---: | ---: |
| `python_engine/tokensmith_cleaning.py` | 145/159 | 91.2% |
| `python_engine/tokensmith_engine.py` | 849/950 | 89.4% |
| `python_engine/tokensmith_preparation_job.py` | 98/112 | 87.5% |
| `python_engine/tokensmith_preparation.py` | 146/162 | 90.1% |
| `python_engine/tokensmith_store.py` | 836/929 | 90.0% |
| `src/main/engine/cloud-generator-service.ts` | 211/211 | 100.0% |
| `src/main/engine/embedding-device.ts` | 41/41 | 100.0% |
| `src/main/engine/engine-service.ts` | 0/177 | 0.0% |
| `src/main/engine/ollama-library-search.ts` | 211/223 | 94.6% |
| `src/main/engine/ollama-service.ts` | 441/1085 | 40.6% |
| `src/main/engine/question-rewrite.ts` | 65/69 | 94.2% |
| `src/main/engine/remote-chat-parameters.ts` | 9/9 | 100.0% |
| `src/main/engine/remote-chat-service.ts` | 304/359 | 84.7% |
| `src/main/engine/remote-generator-network.ts` | 6/7 | 85.7% |
| `src/main/engine/remote-model-secrets.ts` | 62/67 | 92.5% |
| `src/main/engine/study-chat-format.ts` | 856/881 | 97.2% |
| `src/main/engine/study-engine-core.ts` | 86/93 | 92.5% |
| `src/main/index.ts` | 0/598 | 0.0% |
| `src/main/python/python-engine-service.ts` | 200/804 | 24.9% |
| `src/main/system/detect-device-capabilities.ts` | 26/73 | 35.6% |
| `src/main/system/detectors/command.ts` | 2/23 | 8.7% |
| `src/main/system/detectors/electron-gpu.ts` | 32/34 | 94.1% |
| `src/main/system/detectors/linux-gpu-detector.ts` | 52/85 | 61.2% |
| `src/main/system/detectors/macos-gpu-detector.ts` | 99/118 | 83.9% |
| `src/main/system/detectors/nvidia-gpu.ts` | 37/46 | 80.4% |
| `src/main/system/detectors/portable-detector.ts` | 19/50 | 38.0% |
| `src/main/system/detectors/windows-gpu-detector.ts` | 17/37 | 45.9% |
| `src/preload/index.ts` | 0/151 | 0.0% |
| `src/renderer/src/answer-source-preview.ts` | 38/38 | 100.0% |
| `src/renderer/src/AnswerExplanation.tsx` | 0/41 | 0.0% |
| `src/renderer/src/App.tsx` | 0/7398 | 0.0% |
| `src/renderer/src/chat-interactions.ts` | 59/59 | 100.0% |
| `src/renderer/src/ChatDepthPicker.tsx` | 0/53 | 0.0% |
| `src/renderer/src/ChatModelPicker.tsx` | 0/67 | 0.0% |
| `src/renderer/src/CloudGeneratorDialog.tsx` | 0/204 | 0.0% |
| `src/renderer/src/components/DeviceRecommendationPanel.tsx` | 0/97 | 0.0% |
| `src/renderer/src/ConversationViewport.tsx` | 0/231 | 0.0% |
| `src/renderer/src/GenerationStatus.tsx` | 0/27 | 0.0% |
| `src/renderer/src/hooks/useDeviceCapabilities.ts` | 0/51 | 0.0% |
| `src/renderer/src/hooks/useGenerationProgress.ts` | 0/83 | 0.0% |
| `src/renderer/src/hooks/useSourceReader.ts` | 0/52 | 0.0% |
| `src/renderer/src/LibraryWorkspace.tsx` | 0/204 | 0.0% |
| `src/renderer/src/main.tsx` | 0/12 | 0.0% |
| `src/renderer/src/markdown-source.ts` | 107/107 | 100.0% |
| `src/renderer/src/MarkdownSourceViewer.tsx` | 95/150 | 63.3% |
| `src/renderer/src/MessageText.tsx` | 35/36 | 97.2% |
| `src/renderer/src/QuestionEditor.tsx` | 0/63 | 0.0% |
| `src/renderer/src/remark-model-math.ts` | 128/128 | 100.0% |
| `src/renderer/src/SelectedPassagePreview.tsx` | 0/21 | 0.0% |
| `src/renderer/src/source-navigation.ts` | 30/30 | 100.0% |
| `src/renderer/src/SourceNavigation.tsx` | 22/23 | 95.7% |
| `src/renderer/src/ThemePicker.tsx` | 0/50 | 0.0% |
| `src/shared/app-state.ts` | 0/236 | 0.0% |
| `src/shared/bridge.ts` | 0/94 | 0.0% |
| `src/shared/chat-selection.ts` | 21/21 | 100.0% |
| `src/shared/cleaning.ts` | 0/122 | 0.0% |
| `src/shared/cloud-generators.ts` | 51/52 | 98.1% |
| `src/shared/device-capabilities.ts` | 50/50 | 100.0% |
| `src/shared/device-tier-policy.ts` | 69/69 | 100.0% |
| `src/shared/device-tier.ts` | 256/258 | 99.2% |
| `src/shared/embedding-settings.ts` | 7/7 | 100.0% |
| `src/shared/engine.ts` | 0/171 | 0.0% |
| `src/shared/generation-estimate.ts` | 81/81 | 100.0% |
| `src/shared/model-catalog.ts` | 29/34 | 85.3% |
| `src/shared/model-defaults.ts` | 87/87 | 100.0% |
| `src/shared/model-providers.ts` | 52/52 | 100.0% |
| `src/shared/model-recommendation.ts` | 58/60 | 96.7% |
| `src/shared/ollama.ts` | 66/66 | 100.0% |
| `src/shared/preparation.ts` | 51/51 | 100.0% |
| `src/shared/quiz.ts` | 80/82 | 97.6% |
| `src/shared/retired-model-state.ts` | 32/32 | 100.0% |
| `src/shared/retrieval-budget.ts` | 22/22 | 100.0% |
| `src/shared/study-chat-pipeline.ts` | 75/75 | 100.0% |
| `src/shared/typography.ts` | 15/15 | 100.0% |

</details>

Measured commit: ed5ab1415401e2bc45bc6482645ea94f15bc59ae

[CI run](https://github.com/georgia-tech-db/TokenSmith/actions/runs/36469726008)
