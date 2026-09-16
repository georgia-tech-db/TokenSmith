# Local Model Comparison

## Current App Evaluation

The app now uses the selected chat model to resolve conversational questions,
then performs one library search and generates an answer to the original student
question. The old heuristic router has been removed. `question-rewrite.ts` owns
the prompt/schema; `study-chat-pipeline.ts` owns the orchestration. No model name,
course vocabulary, or follow-up wording is hard-coded into routing.

For an organic evaluation, use the rebuilt GUI. Read each actual answer before
choosing the next student question. Record which claim or example prompted it.
Evaluate explicit topic changes separately, rather than calling them follow-ups.
Inspect both `chat_question_rewrite` and `chat_request_context` in View Log.
Malformed resolver output is an error; genuine ambiguity can ask for clarification
without retrieval or a speculative answer. No heuristic fallback is used.

Export the selected conversations and their logged inputs after the GUI run:

```sh
node tests/experiments/export-gui-evaluation.mjs --user-data "$TOKENSMITH_USER_DATA" --ids CONVERSATION_IDS --since ISO_TIMESTAMP --out tmp/model-comparison/gemma-gui-pipeline.json
```

This reads existing results; it does not generate questions or answers. Exported
response times include retrieval, generation and suggested-question calls. The
CI multi-turn suite is explicitly a mocked pipeline-contract check, not an LLM
accuracy score. Existing grounding checks remain separate.

## Historical Experiments

The experiments below predate integration. `compare-question-models.mjs` depended
on the now-deleted heuristic router and is an archived baseline runner; its saved
results remain available. Other replay tools operate on saved inputs and should
not be presented as GUI runs or as the current integrated pipeline.

These opt-in experiments compare the existing conversation router with LLM query
resolution and inspect LLM-generated suggestions without template backfill.
They are not part of normal CI and do not download models automatically.

Requirements: the app's indexed BuzzDBBook collection, its reported rehashing
exchange in the log, local Ollama with `nomic-embed-text`, `llama3:latest`,
`qwen3:1.7b`, and `qwen3:0.6b`, and the bundled Python runtime.

```sh
node tests/experiments/compare-question-models.mjs --user-data "$TOKENSMITH_USER_DATA" --out tmp/model-comparison/report.json
node tests/experiments/compare-rewrite-only.mjs --input tmp/model-comparison/report.json --out tmp/model-comparison/rewrite-only.json
node tests/experiments/replay-rewrites.mjs --user-data "$TOKENSMITH_USER_DATA" --original tmp/model-comparison/report.json --rewrites tmp/model-comparison/rewrite-only.json --out tmp/model-comparison/rewrite-answers.json
```

The library worker copies SQLite through a read-only connection into a temporary
snapshot. Searches use the production hybrid pipeline and a real query embedding
against the stored document embeddings. A run fails if the vector leg has no hits.
The installed app's index and conversations are not modified.

The first run uses one logged answer and five freshly generated answers as fixed
history for an 11-case comparison across the three models. Inputs cover examples,
elaboration, typos, comparisons, short and long topic changes, ambiguous references,
and example identifiers. The second run uses the identical history but asks only
for a rewritten question. Suggestions are generated from those answers. Selected
actual suggestions are then asked through retrieval and answer generation.

JSON output includes raw model responses, prompts/inputs, model details, sources,
and wall-clock timings. Resolver timings follow a warm-up call; model downloads
and first-load latency are not included. Each case is sampled once per variant.

The routing/subject and topic/copy checks are narrow diagnostics, not accuracy
scores. They cannot detect dropped example requests, reversed comparisons,
invented premises, or a generated answer masquerading as a question. Review the
complete responses and retrieved sources before deciding whether to adopt a model.
No LLM resolver is enabled in the app by these experiments.

## Follow-Up Prompt Replay

To replay all original scenarios with the exact prompt, model, schema, and settings
from a saved single-question request:

```sh
node tests/experiments/compare-followup-prompt.mjs --input tmp/model-comparison/report.json --previous tmp/model-comparison/rewrite-only.json --prompt-from tmp/model-comparison/pinning-rewrite-retry.json --out tmp/model-comparison/followup-prompt-all.json
```

This requires the saved inputs and their model in local Ollama. It makes one
warm-up call followed by one rewrite call per case. It does not run retrieval or
answer generation. The output preserves every request and raw response alongside
the previous rewrite for manual comparison; there is no automated accuracy score.

Use `--model` to change only the model, retaining the saved prompt and explicit
generation settings. For a cross-model comparison, pass a single-model replay
report as `--previous`:

```sh
node tests/experiments/compare-followup-prompt.mjs --input tmp/model-comparison/report.json --previous tmp/model-comparison/followup-prompt-all.json --prompt-from tmp/model-comparison/pinning-rewrite-retry.json --model gemma4:e4b --out tmp/model-comparison/gemma4-e4b-followup.json
```

## Answer Quality

`generator-review-cases.json` selects ten saved hybrid-evidence cases and defines
manual review criteria. Criteria are not sent to the generators. The paired run
uses the same current app messages for both models, checks that all four sources
fit without clipping, and saves raw responses plus app-normalized text:

```sh
node tests/experiments/compare-answer-generators.mjs --original tmp/model-comparison/report.json --replays tmp/model-comparison/rewrite-answers.json --out tmp/model-comparison/gemma-answer-quality-fixed.json
```

For a live conversation, `run-study-answer-turn.mjs` appends one turn to a session
file. Choose the next question after reading the actual previous answer. First
turns search directly; subsequent turns use the saved rewrite prompt with the
previous actual question/answer, then fresh production hybrid retrieval and
generation. This is an experimental pipeline, not the installed app router:

```sh
node tests/experiments/run-study-answer-turn.mjs --user-data "$TOKENSMITH_USER_DATA" --session tmp/model-comparison/gemma-cache-live.json --rewrite-from tmp/model-comparison/general-rewrite-request.json --question 'How is 2Q different from LRU?'
```

Source-checked manual assessments can be rendered alongside every actual answer
with `render-answer-comparison.mjs --fixed REPORT --review REVIEW --out MARKDOWN`.
The reported labels are small-sample manual judgments, not automatic answer
accuracy or an independent model judge. None of these scripts changes app settings.
