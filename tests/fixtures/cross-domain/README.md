# Cross-domain benchmark excerpts

Six cases extend the existing accuracy benchmark with reduction direction, input encoding,
compiler-pass minimization, sufficient versus necessary correctness checks, recovery isolation,
and encryption threat models. Cases were adapted from PR #167 and independently checked;
reference answers are evaluation data and are never supplied to the model.

Only 11 PDF pages are included, not the five full documents proposed by #149/#167.
Physical page numbers refer to the exact original files identified by SHA-256 in the
adjacent provenance files. Excerpts and AI transcriptions are adaptations.

| Files | Original work and attribution | Physical pages | Reuse basis |
| --- | --- | --- | --- |
| algorithms-reductions, algorithms-subset-sum | Jeff Erickson, *Algorithms*, chapter 12, NP-Hardness, printed pp. 392–393 and 403–404. [Author's book](https://jeffe.cs.illinois.edu/teaching/algorithms/) | 14–15, 25–26 | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| compiler | Chris Cummins, Volker Seeker, Dejan Grubisic, Baptiste Rozière, Jonas Gehring, Gabriel Synnaeve, Hugh Leather, *Meta Large Language Model Compiler: Foundation Models of Compiler Optimization*. [Paper](https://arxiv.org/abs/2407.02524) | 7–8 | CC BY 4.0 (paper; no model weights included) |
| nist-recovery, nist-encryption | Ramaswamy Chandramouli and Doron Pinhas, *Security Guidelines for Storage Infrastructure*, NIST SP 800-209, October 2020. [Publication](https://doi.org/10.6028/NIST.SP.800-209) | 57–59, 62–63 | NIST publication states it is not subject to US copyright. [NIST policy](https://www.nist.gov/open/license) |

No endorsement by the authors, Meta, or NIST is implied. The repository's software license
does not replace the source documents' terms.

## Preparation and review

The production PDF AI preparation pipeline transcribed page images and selected boundaries,
using local `gemma4:26b`, context 16384, Ollama 0.40.15. No benchmark questions or answers
were passed to preparation. Model digest, prompt hash, original/excerpt hashes, page mapping,
and elapsed time are recorded in each provenance JSON. Preparation took approximately
three minutes total on the development machine; it is **not run in CI**.

The frozen JSON is the retrieval input. Neighboring material is retained as distractors.
Transcriptions are unedited output from the app OCR component. Known errors in the
subset-sum example (missing base-4 digits and an extra summand) are retained, not
manually repaired. Review determines test expectations; it does not rewrite OCR text.
AI boundaries, including page-boundary continuations and occasional footer noise, are retained.
These fixtures test retrieval over actual AI preparation output, not PDF extraction accuracy.

To regenerate from the included PDF excerpts through the app OCR component, run:

```sh
app_runtime/python/bin/python tests/benchmarks/prepare_cross_domain.py /tmp/new-cross-domain
```

Inspect regenerated output against the PDFs, but do not manually correct transcriptions.
Do not silently replace frozen fixtures or expect bit-identical AI output. Rebuild the existing Nomic cache after any fixture edits.
CI caches passage embeddings, embeds each question afresh, and uses real hybrid retrieval.
Each new case uses the seven-source budget calculated by the app for the live-test
profile (8192 context tokens, 1536 answer tokens, configured maxSources 4), scoped to
its collection. Required evidence is unchanged; no artificial top-three cutoff is applied.
A unit check compares these limits with the production budget calculator.
The CI score measures evidence coverage, not generated-answer correctness. The existing
optional live runner can generate these cases using `TOKENSMITH_BENCHMARK_CASES_PATH`.
