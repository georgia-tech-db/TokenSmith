# BuzzDB Book: Basic and AI GUI comparison

Observed September 14, 2026 before the 0.1.13 version bump. This was a native Electron
run with separate fresh profiles, the same full authored BuzzDB Markdown file,
`gemma4:e4b` for answers and AI preparation, and `nomic-embed-text` hybrid retrieval.
Existing model downloads were reused. It was not a clean-machine installer test.

## Method and import

Three seed questions were identical across modes. Follow-ups were chosen only after
reading each new answer. Six questions matched exactly; the final slotted-page
follow-up was adapted to the AI answer. Seven answers per mode were reviewed manually.
One generation per question does not establish a statistically reliable accuracy rate.

- Basic used 1,574 authored chunks. Its GUI import was not precisely timed.
- AI made 348 successfully cached boundary responses and produced 1,266 chunks in
  1,046 linked source units. Reconstructing the chunks preserved the entire source.
- AI initially failed on otherwise valid out-of-order boundary lists. Validation now
  checks integer type and range, then sorts and deduplicates the selected positions.
  Three final responses needed normalization; no source text or boundaries were invented.
- After user-approved resumption, import took 24.89 minutes including retries and
  fixing/testing, in addition to roughly 22 minutes of previous preparation. The final
  post-fix segment took 16.49 minutes. Total wall time was 63.79 minutes including
  interruptions and waiting. These are not uninterrupted performance measurements.
- AI retained metadata comments in 1,039 chunks; 31 units had comment-based titles.
  These cluttered source previews and inflated the displayed word count. Basic already
  understands the authored metadata and handles this input more cleanly.

## Answers

| Question | Basic | AI |
| --- | --- | --- |
| Why do we need to pin a page if the buffer already uses LRU? | Correct | Correct; useful pinning source ranked fourth rather than second |
| What happens if we forget to unpin a page? | Correct but shallow | Clearer explanation of reduced capacity and possible exhaustion |
| How does a slotted page let records move without breaking references to them? | Correct | Correct stable RID, directory, and offset explanation |
| Can you show how moving data updates just one spot? | Repeated the mechanism | Correct symbolic update, still not a numerical demonstration |
| Final slotted-page example follow-up | Drifted to secondary-index lookup | Stayed on topic but invented overlapping records |
| Why process rows in batches instead of one at a time? | Correct | Correct overhead, locality, and SIMD explanation |
| How exactly does SIMD work with data batches? | Misleading horizontal-sum example | Correct sixteen-lane comparison example |

The final AI slotted-page question was: "Can you show a small numerical example of
that RID and offset before and after a record moves?" Its answer retained T1 at offset
50 with length 20, then moved T2 to offset 65. T1 occupies bytes 50-69, so the records
overlap by five bytes. The four sources explained indirection correctly and did not
supply that invented layout. This is a generator arithmetic error, not corrupted text.

All AI answers used four hybrid sources, an effective 8,192-token window, a 1,024-token
answer reserve, and zero source truncations. Estimated prompts were 2,774-4,770 tokens;
saved response times were 12.9-34.7 seconds. Source ordering was still noisy, including
tangential passages ahead of the decisive pinning and batching explanations.

Initial and follow-up suggestions appeared, but some repeated already explained
material or contained poor wording or malformed math. Restart preserved the ready
collection and all seven answer texts in the three test conversations. Startup opened
a new blank chat; the saved conversations remained available in the sidebar.

## Decision and evidence

Keep Basic as default and AI explicitly experimental. This run did not demonstrate an
overall AI advantage for authored BuzzDB Markdown and does not certify PDF grouping,
extraction correction, cloud preparation, or answer faithfulness across collections.

Local full answers, source text, model settings, rewritten queries, and timings remain
in `tmp/buzzdb-basic-first-run` and `tmp/buzzdb-ai-first-run`. These private test profiles,
logs, and generated artifacts are intentionally excluded from Git. The structural
regression tests for boundary normalization, source preservation, and resumability are
included in the Python suite; they are not automatic answer-quality grades.
