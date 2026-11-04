# Proposal implementation plan

The first delivery supports reproducible observations and reviewed assessments inside existing Practice. Keep observations separate from estimates and retain historical unknowns. Use saved session JSON, not a competing mastery database. No new model calls for recording or replay.

1. Add optional course/concept and item/version identifiers, observation timestamps, assistance references, response timing, and assessment session metadata. Old sessions remain readable without fabricated timestamps or concept mappings. New free-form topics get session-local identities; only reviewed IDs aggregate across sessions.
2. Record presentation/resume and pause intervals, first submission, and actual help content. Measure wall time separately from visible time. Serialize and atomically replace saved state. Await submission save before advancing; expose retry after persistence failures. Preserve idempotence and existing deletion semantics.
3. Validate and import reviewed JSON item packs into sessions, without indexing questions or keys into the Library. Support True/False and short explanations. Keep assessment keys out of the question UI and suppress help, sources, quoting, feedback, and answer-bearing navigation until explicit completion. Retain incomplete assessments across restart. Short answers await independent review.
4. Export a versioned session snapshot explicitly. Provide a CLI that emits a blinded response packet separately from the scoring/condition key. Replay uses recorded observations only; retain missing timing, unresolved grading and failures. Do not export model API keys.
5. Compute first-response, recency-weighted, and assistance-aware baselines as pure functions. Group by course/concept IDs, exclude unknown concepts and unresolved feedback, retain sample counts. Accept explicit versioned score corrections for replay without modifying original attempts. Label output practice performance, not calibrated mastery.
6. Add a bounded experimental runner that accepts frozen production requests with explicit evidence variants and direct/reasoning settings. Use the actual Ollama execution path, preserve inputs/results/failures and total timing; no automatic planner, solver expansion, or synthetic claims about learning. Reviewed evidence is labeled a diagnostic condition.

Validation: focused unit tests for import rejection, legacy compatibility, concept isolation, missing data, first-response semantics, assistance/correction replay, export separation, state-save ordering and failure recovery; browser checks for assessment controls and response timing; typecheck, production build, existing relevant tests. Do not rerun expensive model experiments merely to validate deterministic plumbing.

Remaining research work: reviewed course packs, independent scoring, real-device trials and human learning studies. The sample pack is a software fixture, not a reviewed research instrument. External assistance cannot be inferred from lack of observed in-app help. Local assessment controls are not a secure examination environment.

## Implementation map and acceptance checks

| Component | Code | Acceptance check |
| --- | --- | --- |
| Backward-compatible observations and identities | `src/shared/practice.ts` | Old sessions keep unknown timestamps/concepts; reviewed course IDs distinguish identical topic labels. |
| Pack validation, assessment masking, export and replay estimates | `src/shared/practice-study.ts` | Reject malformed or duplicate items; pending assessment exports omit keys; corrections alter derived scores without editing responses. |
| Focus/visibility timing | `src/renderer/src/hooks/usePracticeClock.ts` | Pause on blur/hide, resume on focus, preserve cumulative intervals; flag interrupted intervals as incomplete. |
| Reliable state saving | `src/main/atomic-state.ts`, `src/main/index.ts`, `App.tsx` | Serialize writes before any asynchronous work; replace complete snapshots atomically; advance only after acknowledged save. |
| Practice and assessment UI | `PracticeScreen.tsx`, `PracticeComposer.tsx`, `PracticeQuestionView.tsx` | Import without a model, record both response types, resume after restart, withhold help/feedback, complete explicitly, export locally. |
| Help provenance | `App.tsx` | Opening Chat without a response stays unassisted; visible linked replies retain content and message IDs; do not record a background reply as already shown. |
| Review and corrected replay | `tests/experiments/practice-study.mjs` | Separate opaque review packet from condition/scoring key; keep input checksum and correction reasons. |
| Bounded execution experiment | `tests/experiments/proposal-comparison.mjs` | Dry-run verifies six conditions and excludes evaluator-only fields from requests; live execution records failures and unavailable measurements explicitly. |

The browser regression covers both desktop and mobile for normal Practice, plus a dedicated assessment check for import, response timing, restart, completion, export and failed-save retry. Deterministic tests cover the data boundaries and atomic writer. Live model-quality, memory and learning experiments are intentionally separate from implementation verification.

## Deliberate simplifications

- One session persistence store; no separate mastery database or synchronization service.
- No new dependencies or per-response model calls for observations, scoring baselines or export.
- Free-form generated topics use conservative session-local concept identities; reviewed packs supply reusable course concepts.
- Corrections are explicit replay inputs, not a new correction-management screen.
- Fixed evidence variants are supplied to the comparison runner; it does not pretend to automate evidence construction or measure retrieval/preparation costs.
- No calibrated mastery claim, automatic teaching planner, new solver catalog, or background upload.
