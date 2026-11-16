# Local answer evaluation

This runner produces diagnostic, uncalibrated rubric scores. It does not establish model accuracy or hardware compatibility. Review a representative public or synthetic labeled set before treating its gate as a release criterion. Separate generator and judge models are configurable, but independence alone does not establish judge reliability.

Use only public or synthetic fixtures in this repository. Keep confidential inputs and generated reports outside version control. The runner accepts only loopback Ollama endpoints. Model downloads require explicit `--auto-pull`. Downloads require internet access, though evaluation requests remain local.

`--context-length` (8192) and `--max-tokens` (1024) govern both source formatting and Ollama request limits. Oversized estimated prompts fail rather than intentionally relying on truncation. Index profiles are partitioned by PDF content, installed embedding model digest, endpoint and Python engine contents. `--force-reindex` creates a fresh profile.

Dataset cases may supply `messages` with prior user/assistant turns in the app's message format (`role`, `text`). Both live retrieval and cached-source runs use the app question rewriter. Cached sources remain fixed, so this mode cannot assess retrieval quality. Clarifications require manual review. Reports retain the resolution, sources, answer and judge rationale for inspection. PDF indexing here exercises the engine indexing API, not the complete interactive ingestion UI or Practice.

A passing case requires a valid judge response, sufficient score, a correct verdict, a satisfied core claim and no contradictions. Appropriate abstention is judged against the reference. Dry runs simulate answers and judgments and provide no quality evidence. Same-model generation and judging can share errors. Human calibration and source/citation correctness review remain necessary.

Run the synthetic tests without model downloads or real documents:

```sh
node --test tests/experiments/local-eval-policy.test.mjs tests/experiments/local-eval-runner.test.mjs
```

The runner test starts a temporary loopback mock server and deletes its synthetic input and reports afterward.
