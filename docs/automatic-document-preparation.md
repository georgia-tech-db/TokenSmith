# Automatic document preparation

The Library offers **Basic** and **AI** preparation. Basic is the default for newly
added collections and needs an embedder but no preparation/chat model. Existing saved
AI choices remain AI on Update/Resume. The mode picker is visible before submission;
AI is marked Experimental. Optional AI instructions and document overrides remain.
Details provide Markdown/math previews and source locations.

AI preparation version 15 selects numbered source lines rather than copying opening
quotes. Numbers are local to each request and mapped to immutable document blocks by
the app; headings and text are copied from those blocks, not generated. Validation
rejects non-integers and unknown IDs, then sorts and deduplicates valid boundary
locations without changing the selected source lines. Ollama receives a schema
constrained to the supplied line numbers. Old quote caches are not reused; existing
published indexes are not automatically reprocessed. Large units have linked parts.
A separate worker performs the
job, caches completed boundaries and embeddings, and supports pause/resume. An existing
index stays searchable during preparation and embedding. Publication replaces SQLite
content in a transaction after embeddings succeed.

## Validation and present limitations

- The current local draft has a passing production build and unit checks, but its
  cross-document acceptance gate is **not green**. See
  [the source-unit evaluation](document-unit-evaluation.md) for reproducible checks
  and the distinction between structural tests and real answer quality.
- UI checks exercised default import, optional instructions, document overrides, source
  locations, both themes, and narrow layouts.
- The latest real Gemma/Nomic numbered-line run imported all five document samples
  with exact source coverage and hybrid retrieval, but only Looking-Glass and slides
  passed every grouping check. Poems passed 11/24 checks, the filing 1/2, and calculus
  0/2. This is a grouping regression for poems and the filing compared with version 14,
  despite eliminating the previous invalid-quote failures. AI remains experimental.
- The production numbered-line path also prepared all 50 DB textbook pages: 111 chunks
  in 96 linked units, 29 model calls, no boundary retries, exact source preservation,
  and 123.35 seconds including embedding/indexing. Basic took 5.59 seconds, 165 chunks,
  and zero preparation-model calls. These are single-run import checks, not new
  answer-quality measurements. Local artifacts: `tmp/db50-numbered-production` and
  `tmp/document-units-v15`.
- Source-unit expansion retrieves complete indexed parents within a size limit, then
  model-aware prompt budgeting packs the evidence. This does not guarantee that the
  preparation model chose correct semantic boundaries or that the answer is faithful.
- A full BuzzDB Markdown GUI trial completed after fixing validation of unordered or
  duplicate boundary numbers. Seven real answers and a restart check were reviewed;
  one numerical example was incorrect. See [the GUI comparison](buzzdb-preparation-gui-evaluation.md).
  This does not establish an overall AI advantage over Basic.
- The checks use selected PDF pages, not every page of every document. Full-book
  reliability across document types is not established. Some real follow-ups still drift or give circular
  explanations even when the necessary source text is present.
- PDF extraction uses embedded text. OCR, image understanding, and robust reconstruction
  of complex multi-column reading order are not implemented. The UI warns about image
  pages and pages without extractable text.
- If one document fails while updating an existing collection, the old collection remains
  published. Successful preparation is cached for the next attempt.

This is an experimental implementation suitable for trying the flow. LLM boundary quality
needs further evaluation before treating automatic results as consistently correct.

Current verification: production build, 128 TypeScript unit tests, and 111 Python unit
tests pass. Isolated UI checks cover default Basic, explicit AI, saved AI settings,
new-form reset, no-generator Basic submission, missing-generator AI errors, and narrow
and desktop layouts. Cloud/GGUF request plumbing is stub-tested, not live-provider
certification. GUI trials used separate local profiles and existing Ollama models;
the normal profile and model downloads were not removed or reset. No cloud uploads
were used in those trials.
