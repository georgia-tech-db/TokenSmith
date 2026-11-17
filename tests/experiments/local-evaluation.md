# Local answer evaluation

This runner produces diagnostic, uncalibrated rubric scores. It does not establish model accuracy or hardware compatibility. Review a representative public or synthetic labeled set before treating its gate as a release criterion. Separate generator and judge models are configurable, but independence alone does not establish judge reliability.

Use only public or synthetic fixtures in this repository. Keep confidential inputs and generated reports outside version control. The runner accepts only loopback Ollama endpoints. Model downloads require explicit `--auto-pull`. Downloads require internet access, though evaluation requests remain local.

`--context-length` (8192), `--max-tokens` (1024 for generation), and `--judge-max-tokens` (2048) govern source formatting and Ollama request limits. Thinking is disabled when the installed model advertises that capability so structured output has its own predictable budget. Oversized estimated prompts fail rather than intentionally relying on truncation. Index profiles are partitioned by PDF content, installed embedding model digest, endpoint and Python engine contents. `--force-reindex` creates a fresh profile.

Dataset cases may supply `messages` with prior user/assistant turns in the app's message format (`role`, `text`). Both live retrieval and cached-source runs use the app question rewriter. Cached sources remain fixed, so this mode cannot assess retrieval quality. Clarifications require manual review. Reports retain the resolution, sources, answer and judge rationale for inspection. PDF indexing here exercises the engine indexing API, not the complete interactive ingestion UI or Practice.

Cases may provide `requiredFacts`, an array of 1–6 objects with `requirement` and an exact supporting `referenceQuote` from the reference answer. Prefer these human-reviewed rubrics for release gates: reference prose often mixes essential facts and optional explanation. Invalid or empty authored rubrics require manual review. Without an authored rubric, the evaluator derives essential requirements from the question and reference, without seeing the candidate. Inferred requirements can be too strict or too permissive and need calibration. A second call assesses coverage. A separate claim audit checks for false additional statements, even when the main answer is correct. A final verification call independently rechecks any flagged issue, accepting equivalent wording and excluding optional reference details. Candidate and evidence passages have stable IDs, so formatting differences do not invalidate copied quotes. Requirements still need a supporting reference quote. Invalid evidence IDs or unresolved assessments yield manual review. The same judge model performs these stages, so this verification is not an independent ground truth.

Scores are the fraction of applicable requirements covered, not confidence probabilities. A passing case requires all essential facts, a correct verdict, and no verified material errors. Extra errors can fail an answer even with full coverage. Missing requirements and contradictions are reported separately. Reports retain every judge prompt and response for auditing. Appropriate abstention is judged against the reference. Dry runs simulate answers and judgments and provide no quality evidence. Same-model generation and judging can share errors. Human calibration and source/citation correctness review remain necessary.

Run the synthetic tests without model downloads or real documents:

```sh
node --test tests/experiments/local-eval-policy.test.mjs tests/experiments/local-eval-judge.test.mjs tests/experiments/local-eval-runner.test.mjs
```

The runner test starts a temporary loopback mock server and deletes its synthetic input and reports afterward.

## Grading in the existing live BuzzDB benchmark

```sh
npm run test:eval:buzzdb
```

This is an alias for `node scripts/python-dev.mjs benchmark --live --graded`. It runs the existing retrieval benchmark and conversation contracts, then the existing live runner (`tests/benchmarks/buzzdb_reasoning.mjs`) with its pinned grading profile. There is no separate app/Electron runner. `npm run test:benchmark -- --live` retains the broader, ungraded diagnostic mode.

Prerequisites are installed Node dependencies, bundled Python (`npm run setup:python-runtime`), and local Ollama with the model digests in `tests/fixtures/buzzdb/regression/suite.json`. Generation uses Llama 3, embeddings use Nomic, and judging uses Gemma 4 26B. The larger judge requires enough local memory. No models are downloaded by the benchmark. It reads only the public BuzzDB fixture and synthetic calibration controls, never the app library. No display or Electron app launch is needed.

The existing Python benchmark worker builds an isolated index from the public marked Markdown book and reuses its validated embedding cache when available. The live runner calls production rewriting, packing, hybrid retrieval, and generation. Follow-ups use actual generated answers. References, required facts, and judge output never enter the generator conversation. This tests the answer pipeline, not app ingestion, PDF OCR, UI, or Practice.

The graded profile has three normal turns and the same three turns under a deliberate front/back reversal in retrieved passages. Original passages and reference evidence remain intact. Five synthetic controls test both false acceptance and false rejection by the judge. All judgments and calibration results appear in the existing live report, along with original and corrupted retrieval, packed evidence, model digests, settings, and source hashes.

The process exits nonzero for failures in the original retrieval/contracts checks, new incorrect normal answers, unresolved judgments, failed calibration, missing turns, ineffective fault injection, or execution errors. It exits zero only when all applicable gates pass. Correct recovery under corrupted retrieval is allowed. The documented `list-ends` baseline error remains visible as a known failure, not counted as a correct answer. Fixing it is an improvement. New errors within an already-failing case require manual review.

Local reports are stored under ignored `tmp/buzzdb-reasoning-*`: `summary.txt`, `answers.md`, and `results.json`. The expensive live run is not part of CI, and these reports are not uploaded. A green regression gate does not mean every answer is correct. Acceptance counts include deliberate fault cases and must not be interpreted as overall model accuracy.

The graded profile also pins Ollama 0.40.15, the runtime used for its local calibration. The ordinary retrieval CI benchmark retains its separate 0.31.1 runtime pin.

Book/model/runtime identity changes require explicit review and recalibration of the profile. The generator uses temperature zero and its production seed behavior. The pinned seed applies to the judge. These are semantic checks, not promises of bit-for-bit reproducibility across hardware. Add evidence-backed cases for new failures and remove known-failure allowances only after confirming fixes.

## Fast CI checks

```sh
npm run test:eval:fast
```

The `Fast evaluator checks` step within the `Build and tests` job in `.github/workflows/ci.yml` runs on pull requests, main pushes, and merge queues. It uses Node and a temporary synthetic loopback server, with no Ollama, model downloads, Python setup, or app launch. It covers rubric validation, false-accept/false-reject handling, model/book pinning, fault injection, known-failure accounting, missing-run detection, and the live-runner grading/report integration. The expensive BuzzDB model run remains local. These checks share the existing build/test job and its required status.
