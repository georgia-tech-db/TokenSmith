## Unit-Test Line Coverage

| Scope | Covered / total lines | Coverage |
| --- | ---: | ---: |
| Frontend / Electron (TypeScript / TSX) | 7121/19741 | 36.1% |
| Backend (Python) | 2364/2602 | 90.9% |

Includes untested files in `src/` and `python_engine/`; excludes TypeScript declaration files.
TypeScript coverage includes the UI, Electron main/preload, and shared modules, not only the renderer.
Measured with c8 (V8 line coverage) and coverage.py (executable Python lines).
This measures unit-test execution, not answer accuracy or full end-to-end coverage.

<details><summary>Per-file coverage</summary>

| File | Covered / total lines | Coverage |
| --- | ---: | ---: |
| `python_engine/tokensmith_cleaning.py` | 145/159 | 91.2% |
| `python_engine/tokensmith_collections.py` | 33/37 | 89.2% |
| `python_engine/tokensmith_embedding_limits.py` | 1/1 | 100.0% |
| `python_engine/tokensmith_engine.py` | 938/1038 | 90.4% |
| `python_engine/tokensmith_pdf_preparation.py` | 86/86 | 100.0% |
| `python_engine/tokensmith_preparation_job.py` | 107/118 | 90.7% |
| `python_engine/tokensmith_preparation.py` | 162/179 | 90.5% |
| `python_engine/tokensmith_store.py` | 892/984 | 90.7% |
| `src/main/atomic-state.ts` | 17/17 | 100.0% |
| `src/main/collections/book-preparation.ts` | 56/56 | 100.0% |
| `src/main/collections/collection-registry.ts` | 202/211 | 95.7% |
| `src/main/collections/collection-routing.ts` | 35/39 | 89.7% |
| `src/main/engine/cloud-generator-service.ts` | 262/262 | 100.0% |
| `src/main/engine/embedding-device.ts` | 41/41 | 100.0% |
| `src/main/engine/engine-service.ts` | 0/187 | 0.0% |
| `src/main/engine/model-context-metadata.ts` | 90/90 | 100.0% |
| `src/main/engine/model-loading.ts` | 70/70 | 100.0% |
| `src/main/engine/ollama-library-search.ts` | 211/223 | 94.6% |
| `src/main/engine/ollama-service.ts` | 640/1257 | 50.9% |
| `src/main/engine/question-rewrite.ts` | 75/75 | 100.0% |
| `src/main/engine/remote-chat-parameters.ts` | 9/9 | 100.0% |
| `src/main/engine/remote-chat-service.ts` | 317/369 | 85.9% |
| `src/main/engine/remote-generator-network.ts` | 6/7 | 85.7% |
| `src/main/engine/remote-model-secrets.ts` | 62/67 | 92.5% |
| `src/main/engine/study-chat-format.ts` | 882/906 | 97.4% |
| `src/main/engine/study-engine-core.ts` | 86/93 | 92.5% |
| `src/main/engine/study-engine-latency.ts` | 65/65 | 100.0% |
| `src/main/index.ts` | 0/638 | 0.0% |
| `src/main/python/python-engine-service.ts` | 224/857 | 26.1% |
| `src/main/system/detect-device-capabilities.ts` | 26/73 | 35.6% |
| `src/main/system/detectors/command.ts` | 2/23 | 8.7% |
| `src/main/system/detectors/electron-gpu.ts` | 32/34 | 94.1% |
| `src/main/system/detectors/linux-gpu-detector.ts` | 52/85 | 61.2% |
| `src/main/system/detectors/macos-gpu-detector.ts` | 99/118 | 83.9% |
| `src/main/system/detectors/nvidia-gpu.ts` | 37/46 | 80.4% |
| `src/main/system/detectors/portable-detector.ts` | 19/50 | 38.0% |
| `src/main/system/detectors/windows-gpu-detector.ts` | 17/37 | 45.9% |
| `src/preload/index.ts` | 0/165 | 0.0% |
| `src/renderer/src/answer-source-preview.ts` | 38/38 | 100.0% |
| `src/renderer/src/AnswerExplanation.tsx` | 0/73 | 0.0% |
| `src/renderer/src/App.tsx` | 0/6440 | 0.0% |
| `src/renderer/src/chat-interactions.ts` | 93/93 | 100.0% |
| `src/renderer/src/ChatDepthPicker.tsx` | 0/53 | 0.0% |
| `src/renderer/src/ChatModelPicker.tsx` | 0/67 | 0.0% |
| `src/renderer/src/CloudGeneratorDialog.tsx` | 0/251 | 0.0% |
| `src/renderer/src/CollectionLibrary.tsx` | 0/87 | 0.0% |
| `src/renderer/src/components/DeviceRecommendationPanel.tsx` | 0/97 | 0.0% |
| `src/renderer/src/ConversationViewport.tsx` | 0/252 | 0.0% |
| `src/renderer/src/GenerationStatus.tsx` | 0/28 | 0.0% |
| `src/renderer/src/hooks/useDeviceCapabilities.ts` | 0/51 | 0.0% |
| `src/renderer/src/hooks/useGenerationProgress.ts` | 0/28 | 0.0% |
| `src/renderer/src/hooks/usePracticeClock.ts` | 0/39 | 0.0% |
| `src/renderer/src/hooks/useSourceReader.ts` | 0/57 | 0.0% |
| `src/renderer/src/hooks/useStudyDocuments.ts` | 0/36 | 0.0% |
| `src/renderer/src/latency-plan.ts` | 243/247 | 98.4% |
| `src/renderer/src/library-documents.ts` | 0/38 | 0.0% |
| `src/renderer/src/LibraryWorkspace.tsx` | 0/221 | 0.0% |
| `src/renderer/src/main.tsx` | 0/12 | 0.0% |
| `src/renderer/src/markdown-source.ts` | 107/107 | 100.0% |
| `src/renderer/src/MarkdownSourceViewer.tsx` | 95/150 | 63.3% |
| `src/renderer/src/MasteryMeter.tsx` | 0/25 | 0.0% |
| `src/renderer/src/MessageAvatar.tsx` | 0/8 | 0.0% |
| `src/renderer/src/MessageComposer.tsx` | 0/37 | 0.0% |
| `src/renderer/src/MessageText.tsx` | 47/50 | 94.0% |
| `src/renderer/src/practice-client.ts` | 91/91 | 100.0% |
| `src/renderer/src/PracticeChatSuggestions.tsx` | 0/57 | 0.0% |
| `src/renderer/src/PracticeComposer.tsx` | 0/39 | 0.0% |
| `src/renderer/src/PracticeFeedbackView.tsx` | 0/26 | 0.0% |
| `src/renderer/src/PracticeQuestionView.tsx` | 0/57 | 0.0% |
| `src/renderer/src/PracticeScreen.tsx` | 0/426 | 0.0% |
| `src/renderer/src/PracticeTopicPicker.tsx` | 0/135 | 0.0% |
| `src/renderer/src/QueryPlanPanel.tsx` | 0/245 | 0.0% |
| `src/renderer/src/QuestionEditor.tsx` | 0/63 | 0.0% |
| `src/renderer/src/remark-model-math.ts` | 128/128 | 100.0% |
| `src/renderer/src/SelectedPassagePreview.tsx` | 0/22 | 0.0% |
| `src/renderer/src/source-display.ts` | 0/18 | 0.0% |
| `src/renderer/src/source-navigation.ts` | 30/30 | 100.0% |
| `src/renderer/src/SourceNavigation.tsx` | 22/23 | 95.7% |
| `src/renderer/src/SourceTray.tsx` | 0/80 | 0.0% |
| `src/renderer/src/StudyDocumentPicker.tsx` | 0/44 | 0.0% |
| `src/renderer/src/StudyHeader.tsx` | 0/35 | 0.0% |
| `src/renderer/src/StudyMaterialsPane.tsx` | 0/36 | 0.0% |
| `src/renderer/src/StudyModeTabs.tsx` | 0/10 | 0.0% |
| `src/renderer/src/StudySidebar.tsx` | 0/59 | 0.0% |
| `src/renderer/src/ThemePicker.tsx` | 0/50 | 0.0% |
| `src/shared/app-state.ts` | 0/262 | 0.0% |
| `src/shared/bridge.ts` | 0/107 | 0.0% |
| `src/shared/chat-model-selection.ts` | 38/38 | 100.0% |
| `src/shared/chat-selection.ts` | 21/21 | 100.0% |
| `src/shared/cleaning.ts` | 0/122 | 0.0% |
| `src/shared/cloud-generators.ts` | 97/100 | 97.0% |
| `src/shared/collections.ts` | 27/30 | 90.0% |
| `src/shared/device-capabilities.ts` | 50/50 | 100.0% |
| `src/shared/device-tier-policy.ts` | 69/69 | 100.0% |
| `src/shared/device-tier.ts` | 256/258 | 99.2% |
| `src/shared/embedding-settings.ts` | 7/7 | 100.0% |
| `src/shared/engine.ts` | 0/185 | 0.0% |
| `src/shared/generation-estimate.ts` | 92/92 | 100.0% |
| `src/shared/generation-progress.ts` | 92/94 | 97.9% |
| `src/shared/latency-trace.ts` | 358/362 | 98.9% |
| `src/shared/model-catalog.ts` | 29/34 | 85.3% |
| `src/shared/model-context.ts` | 62/66 | 93.9% |
| `src/shared/model-defaults.ts` | 100/100 | 100.0% |
| `src/shared/model-loading.ts` | 0/11 | 0.0% |
| `src/shared/model-providers.ts` | 52/52 | 100.0% |
| `src/shared/model-recommendation.ts` | 58/60 | 96.7% |
| `src/shared/ollama.ts` | 66/66 | 100.0% |
| `src/shared/practice-evidence.ts` | 29/29 | 100.0% |
| `src/shared/practice-mastery.ts` | 94/94 | 100.0% |
| `src/shared/practice-question-plan.ts` | 41/42 | 97.6% |
| `src/shared/practice-study.ts` | 117/117 | 100.0% |
| `src/shared/practice.ts` | 224/228 | 98.2% |
| `src/shared/preparation.ts` | 51/51 | 100.0% |
| `src/shared/quiz.ts` | 114/114 | 100.0% |
| `src/shared/reasoning.ts` | 20/20 | 100.0% |
| `src/shared/retired-model-state.ts` | 31/31 | 100.0% |
| `src/shared/retrieval-budget.ts` | 23/23 | 100.0% |
| `src/shared/saved-topics.ts` | 18/18 | 100.0% |
| `src/shared/study-chat-pipeline.ts` | 95/95 | 100.0% |
| `src/shared/study-scope.ts` | 49/49 | 100.0% |
| `src/shared/topic-inventory-generation.ts` | 78/78 | 100.0% |
| `src/shared/topic-inventory.ts` | 108/108 | 100.0% |
| `src/shared/true-false-statement.ts` | 12/14 | 85.7% |
| `src/shared/typography.ts` | 15/15 | 100.0% |

</details>

Measured commit: bba150df4d5dcec0765a59060b84ff881bb54faf

[CI run](https://github.com/georgia-tech-db/TokenSmith/actions/runs/43544983073)
