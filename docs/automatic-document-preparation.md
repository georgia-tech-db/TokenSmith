# Automatic document preparation

The Library now supports automatic LLM preparation, optional collection instructions,
and per-document overrides. Users can add documents without writing instructions or
approving a review step. Details provide Markdown/math previews and source locations.
Basic preparation remains an explicit alternative.

Preparation selects boundaries in immutable extracted source spans. Validation checks
coverage and ordering; large units have linked parts. A separate worker performs the
job, caches completed boundaries and embeddings, and supports pause/resume. An existing
index stays searchable during preparation and embedding. Publication replaces SQLite
content in a transaction after embeddings succeed.

## Validation and present limitations

- Production build and TypeScript checks pass; 121 TypeScript and 91 Python tests pass.
- UI checks exercised default import, optional instructions, document overrides, source
  locations, both themes, and narrow layouts.
- A local Gemma/Nomic smoke test completed preparation, embedding, storage, and retrieval.
  However, Gemma split a short mixed-content example into individual lines despite the
  grouping instructions. Structural validation guarantees coverage, not semantic quality.
- Representative PDF trials included poetry, a paper, a report, slides, and calculus.
  Some trials preserved complete source coverage, but longer requests encountered timeouts
  and truncated JSON. Request windows were subsequently reduced; the complete five-file
  trial has not passed with the final settings. Full-book reliability is not established.
- PDF extraction uses embedded text. OCR, image understanding, and robust reconstruction
  of complex multi-column reading order are not implemented. The UI warns about image
  pages and pages without extractable text.
- If one document fails while updating an existing collection, the old collection remains
  published. Successful preparation is cached for the next attempt.

This is an experimental implementation suitable for trying the flow. LLM boundary quality
needs further evaluation before treating automatic results as consistently correct.
