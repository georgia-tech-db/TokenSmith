# Practice records and proposal experiments

## In the app

Use normal Practice to collect generated questions and first responses. Saved collection topics supply stable collection-scoped concept IDs across sessions. Free-form titles are never automatically treated as the same concept. Reviewed packs can also supply explicit concept IDs.

In a new Practice session, choose **Import reviewed assessment** and select a version 1 JSON pack. The file shape is illustrated by `tests/fixtures/assessment-pack.json`; that file is a software fixture, not a validated study instrument. Each item has an ID/revision, course-scoped concept, format, reviewed criteria and explanation, plus a boolean key for True/False. Optional `references` are source-locator strings retained in the item record. Import rejects duplicate items, missing criteria and malformed keys. The importer cannot establish that a claimed reviewed key is scientifically correct: authors must review it independently.

Assessments need no ready model or open library. Responses are saved before advancing. Feedback, source/help controls, quoting, and keys are withheld until **Complete assessment**. Short explanations remain **Awaiting review**. An incomplete assessment resumes after restart, and its response-only export omits keys and rubrics. These controls support an assessment protocol; they are not a secure exam system and cannot rule out help from another app, Chat, or another person.

**Export practice record**, beside **Practice another topic**, saves a versioned JSON record with an optional participant code. Export is explicit, local, and contains answers, course content and observed help; inspect it before sharing. Provider credentials, unrelated chats, and local document paths are excluded. Deleting a session removes its conversation and nested observations from future transcript exports. Topic mastery is retained separately. Previously exported files remain independent copies.

**View mastery**, immediately after Export, opens the current topic's mastery estimate and supporting answer history. **Return to practice** brings you back to the same question.

## Observation semantics

- First responses retain their text, timestamp, versioned item/key identity and observed assistance IDs.
- Wall time begins at the first recorded presentation. Visible time sums focused, visible intervals, measured with a monotonic clock.
- A crash/reload may lose the tail of an interval; `timingComplete: false` marks this rather than fabricating a complete duration. Old attempts have no new observation version and retain unknown timing/concept provenance.
- Opening a help chat alone does not count as help. An assistant response displayed in the linked active chat records its text and message reference. This establishes delivery, not that the student read or understood it. Absence of observed help does not establish unaidedness.
- Source text is retained with generated items; `generationMs` records question preparation time when available. No missing model digest, collection revision, memory measurement, or historical event is fabricated.
- Saved snapshots are serialized and atomically replaced. A failed submission save exposes retry and does not advance. This does not promise preservation of input that has not yet been acknowledged when the application is forcibly killed.

## Blinded review and replay

From the repository root:

```sh
node tests/experiments/practice-study.mjs --input exported-record.json --out /tmp/new-study-review
```

This makes no model calls. `review.json` contains shuffled opaque identifiers, questions, criteria (when available), and first responses. Keep `key.json`, the original export, and `summary.json` away from reviewers until grading is complete. Text itself can reveal condition clues; the export removes identifying metadata, not those semantic clues.

The summary computes three deliberately simple baselines:

1. `first_response`: mean first-response correctness.
2. `recency`: exponentially weighted correctness with factor 0.9 per observation.
3. `assistance`: the same decay, with observed assisted responses receiving half the observation weight, not an automatic incorrect half-answer.

These are practice-performance estimators, not calibrated mastery probabilities. They exclude incomplete assessments, ungraded items and records without stable concepts/new timestamps. The CLI produces all three for experiments; the student-facing topic view uses the separate estimate below.

## Student-facing topic mastery

The chapter/topic picker and active Practice header replay saved answers through `src/shared/practice-mastery.ts`. Topic rows show an estimate and a bar; objectives, sources and scoring details expand on demand. Unassessed topics show **Not practiced**, not an inferred failure. The chapter bar averages all child topic estimates, with unassessed topics contributing zero, and shows coverage separately. It is a progress summary, not a chapter-level probability. **Practice next** chooses the lowest estimate, including unassessed topics.

Only a first, independently answered True/False question with a versioned topic identity contributes. The current experiment generates the statement, key, explanation and supporting quote in one call, without a verifier or automatic repair. Generated records carry `masteryCheck.verification: 'none'`. The score uses the model's key, so an incorrect key can incorrectly reward or penalize the learner. Exact statements and identical quoted evidence are deduplicated when replaying mastery; this does not prevent repeats from appearing in Practice.

Saved topics use their canonical mapped passages, scoped to the selected documents, without a fresh topic-title search. Each session allows two generated questions per distinct, nonempty mapped passage: one passage allows two questions, ten allow twenty. Each request advances a persisted passage cursor, continuing across sessions for the same collection/topic, and skips passages already used twice in the current session. Failed or cancelled requests consume no question slot. After the final answer, Practice shows completion and offers another topic without another generation call. This is a session limit, not a claim that the topic is mastered; starting another session permits deliberate review.

A new question requests True or False with an independent 50/50 draw; a manual Retry retains that target and advances to the next eligible passage. The prompt includes up to 12 recent topic questions plus up to 12 prior questions from the selected passage, even if they fell outside that topic-history window. Same-passage history includes the generated keys, tested ideas and supporting quotes, but no student responses. History survives restarting while its conversations remain saved; clearing history removes those questions while retaining mastery. Parsing checks the required data structure only: no quote, topic, wording, novelty or truth-target gates. The prompt discourages restatements and truth-value flips, but does not guarantee semantic novelty. If the model returns a different key from the requested target, both are saved; the app does not silently change the key. This means the requested distribution is random, but the model's actual distribution may differ.

Version 1 uses prior `0.2`, guessing `0.5`, slip `0.1`, retention `0.9`. Starting with evidence `E = 0`, an eligible answer updates `E = 0.9 * E + log(0.9 / 0.5)` when correct or `E = 0.9 * E + log(0.1 / 0.5)` when incorrect. The estimate is `sigmoid(logit(0.2) + E)`. Correct, correct, incorrect, correct, correct gives approximately **31%, 43%, 12%, 21%, 32%**. These are provisional assumptions, not calibrated student probabilities or grades.

Help, revisions and repeated claims neither change nor decay the estimate. There is no time-based forgetting rule. Records without the versioned topic identity, short answers and reviewed assessments do not feed this generated-question estimate. Reading supporting material, manually or through Read while waiting, does not count as help. Sources appear in the right pane before answering when Show Sources is enabled. Only actual Chat help before answering produces **Answered with help · mastery unchanged**. Loading saved conversations removes earlier source-view assistance flags and recomputes mastery from those conversations; archived records without the original assistance details cannot be reclassified.

Deleting one conversation or clearing Practice history preserves compact learning records under `practice.masteryRecords`, in the same atomic saved-state snapshot as the deletion. These keep collection/topic identity, first-response correctness, assistance, timestamps and fingerprints for duplicate detection. They do not keep questions, answers, quoted passages or chat text. Mastery replays these records together with retained conversations in chronological order, deduplicating the same response and repeated claims. Scores, coverage and filters therefore survive deletion and restart, and later answers continue from the existing estimate. Manual review and transcript export remain available for conversations still in history; exported files do not include records from deleted conversations.

Run the one-call diagnostic against a prepared BuzzDBBook inventory:

```sh
PRACTICE_INVENTORY=/path/to/topic-inventory.json node tests/experiments/practice-mastery.mjs
```

It generates 16 questions with `gemma4:e4b` by default (override with `PRACTICE_MODEL`), saves prompts, passages, outputs and timings to `/tmp/tokensmith-practice-single-call.json`, and never changes the app profile. `PRACTICE_TOPICS` can select other saved topic titles and counts. Quote matches, repeated text and requested-key mismatches are diagnostic measurements only; correctness and answerability still need human review against the logged passages.

To replay independently reviewed corrections without changing original attempts, supply a JSON array:

```json
[{"sessionId":"session-id","questionId":"question-id","keyVersion":"review-2","correct":true,"reason":"Independent reviewer accepted an equivalent explanation."}]
```

```sh
node tests/experiments/practice-study.mjs --input exported-record.json --corrections corrections.json --out /tmp/new-corrected-review
```

The output preserves the correction provenance and input checksum. Binary correctness is only a baseline summary; the original explanation and rubric remain available for richer independent scoring.

## Fixed-evidence execution comparison

`tests/experiments/proposal-comparison.mjs` compares three supplied evidence variants (`baseline`, `expanded`, `reviewed`) under reasoning off/on. Each frozen manifest has `version: 1` and `cases`, each with a unique `id`, `prompt`, Ollama `model`, `modelSettings`, and `evidence` mapping each variant to a nonempty array of production `ChatSource` objects. It deliberately does not read evaluator keys into the request. Set matching context/output budgets and sampling settings in the manifest.

```sh
node tests/experiments/proposal-comparison.mjs --input cases.json --out /tmp/new-request-inspection --dry-run
./node_modules/.bin/electron tests/experiments/proposal-comparison.mjs --input cases.json --out /tmp/new-comparison
```

The live runner uses production source packing and Ollama generation in a temporary app profile, captures available model digests/residency and source hashes, interleaves condition order by case, and saves failures alongside successes after every request. It never downloads models. Keep the recorded model digests when reproducing a run.

This is an answer-stage comparison with frozen evidence, **not** an end-to-end retrieval benchmark. Reviewed evidence is a diagnostic ceiling. Retrieval/preparation time and process peak memory are explicitly unavailable, not zero; measure them separately using the retrieval/preparation runners before making full-cost or device-access claims. One pass is not a latency distribution: run fresh directories with repeated/interleaved trials on the target devices. No automatic planner or new solver library is included.

## Validation commands

Run the unit tests and `npm run build`. The browser fixture uses deterministic replies and isolated session storage:

```sh
npx vite --config tests/ui/vite.config.mjs --host 127.0.0.1 --port 5188
node tests/ui/practice-assessment-check.mjs
```

Set `PLAYWRIGHT_MODULE` and `CHROME_PATH` if Playwright/Chromium are installed outside the repository. The browser test checks key/help withholding, response timing, reload, short answers, completion, export, and failed-save retry without model calls.

`node tests/experiments/practice-mastery.mjs` is an opt-in live diagnostic using installed Ollama `gemma4:e4b` (override with `PRACTICE_MODEL`). It checks topic fit and repeated claims on fixed BuzzDB passages through the production provider, then generates and scores one question with a scripted answer. It saves model replies to `/tmp/tokensmith-mastery-live.json` (override with `PRACTICE_OUTPUT`) without changing the app's history. It tests question validation and score wiring, not retrieval accuracy or the calibration of mastery estimates.

Run `node tests/integration/practice-comparison.test.mjs` on a desktop with Electron available to check the live runner against a temporary local test server. It verifies startup, all six conditions, both reasoning settings, and continued result recording after an injected failure. It uses no real model or user profile; it does not measure model quality or inference latency.
