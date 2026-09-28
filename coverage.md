## Unit-Test Line Coverage

| Scope | Covered / total lines | Coverage |
| --- | ---: | ---: |
| Frontend / Electron (TypeScript / TSX) | 3948/15590 | 25.3% |
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
| `src/main/engine/engine-service.ts` | 0/175 | 0.0% |
| `src/main/engine/ollama-library-search.ts` | 211/223 | 94.6% |
| `src/main/engine/ollama-service.ts` | 386/1010 | 38.2% |
| `src/main/engine/question-rewrite.ts` | 55/59 | 93.2% |
| `src/main/engine/remote-chat-parameters.ts` | 9/9 | 100.0% |
| `src/main/engine/remote-chat-service.ts` | 299/354 | 84.5% |
| `src/main/engine/remote-generator-network.ts` | 6/7 | 85.7% |
| `src/main/engine/remote-model-secrets.ts` | 62/67 | 92.5% |
| `src/main/engine/study-chat-format.ts` | 815/841 | 96.9% |
| `src/main/engine/study-engine-core.ts` | 84/91 | 92.3% |
| `src/main/index.ts` | 0/574 | 0.0% |
| `src/main/models/local-model-service.ts` | 0/35 | 0.0% |
| `src/main/python/python-engine-service.ts` | 196/803 | 24.4% |
| `src/main/system/detect-device-capabilities.ts` | 26/73 | 35.6% |
| `src/main/system/detectors/command.ts` | 2/23 | 8.7% |
| `src/main/system/detectors/electron-gpu.ts` | 32/34 | 94.1% |
| `src/main/system/detectors/linux-gpu-detector.ts` | 52/85 | 61.2% |
| `src/main/system/detectors/macos-gpu-detector.ts` | 99/118 | 83.9% |
| `src/main/system/detectors/nvidia-gpu.ts` | 37/46 | 80.4% |
| `src/main/system/detectors/portable-detector.ts` | 19/50 | 38.0% |
| `src/main/system/detectors/windows-gpu-detector.ts` | 17/37 | 45.9% |
| `src/preload/index.ts` | 0/143 | 0.0% |
| `src/renderer/src/AnswerExplanation.tsx` | 0/46 | 0.0% |
| `src/renderer/src/App.tsx` | 0/7461 | 0.0% |
| `src/renderer/src/chat-interactions.ts` | 61/61 | 100.0% |
| `src/renderer/src/ChatDepthPicker.tsx` | 0/53 | 0.0% |
| `src/renderer/src/ChatModelPicker.tsx` | 0/67 | 0.0% |
| `src/renderer/src/CloudGeneratorDialog.tsx` | 0/204 | 0.0% |
| `src/renderer/src/components/DeviceRecommendationPanel.tsx` | 0/126 | 0.0% |
| `src/renderer/src/ConversationViewport.tsx` | 0/216 | 0.0% |
| `src/renderer/src/hooks/useDeviceCapabilities.ts` | 0/53 | 0.0% |
| `src/renderer/src/LibraryWorkspace.tsx` | 0/200 | 0.0% |
| `src/renderer/src/main.tsx` | 0/12 | 0.0% |
| `src/renderer/src/markdown-source.ts` | 87/87 | 100.0% |
| `src/renderer/src/MarkdownSourceViewer.tsx` | 75/105 | 71.4% |
| `src/renderer/src/MessageText.tsx` | 35/36 | 97.2% |
| `src/renderer/src/QuestionEditor.tsx` | 0/54 | 0.0% |
| `src/renderer/src/remark-model-math.ts` | 128/128 | 100.0% |
| `src/renderer/src/ThemePicker.tsx` | 0/50 | 0.0% |
| `src/shared/app-state.ts` | 0/233 | 0.0% |
| `src/shared/bridge.ts` | 0/92 | 0.0% |
| `src/shared/cleaning.ts` | 0/122 | 0.0% |
| `src/shared/cloud-generators.ts` | 51/52 | 98.1% |
| `src/shared/device-capabilities.ts` | 50/50 | 100.0% |
| `src/shared/device-tier-policy.ts` | 69/69 | 100.0% |
| `src/shared/device-tier.ts` | 256/258 | 99.2% |
| `src/shared/engine.ts` | 0/160 | 0.0% |
| `src/shared/model-catalog.ts` | 33/38 | 86.8% |
| `src/shared/model-defaults.ts` | 87/87 | 100.0% |
| `src/shared/model-providers.ts` | 52/52 | 100.0% |
| `src/shared/model-recommendation.ts` | 58/60 | 96.7% |
| `src/shared/ollama.ts` | 66/66 | 100.0% |
| `src/shared/preparation.ts` | 50/50 | 100.0% |
| `src/shared/quiz.ts` | 80/82 | 97.6% |
| `src/shared/retrieval-budget.ts` | 21/21 | 100.0% |
| `src/shared/study-chat-pipeline.ts` | 71/71 | 100.0% |

</details>

Measured commit: 454d54cd0396ff692c529b9194074823a50bbed4

[CI run](https://github.com/georgia-tech-db/TokenSmith/actions/runs/36439122948)
