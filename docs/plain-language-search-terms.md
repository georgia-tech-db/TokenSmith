# Plain-language search terms

Optional preparation step. For each passage it writes one sentence in the everyday words a student
would use, stores it as that chunk's search terms, and indexes and embeds it with the passage. The
sentence is never added to the passage text, never shown in chat, and never enters the answer window.
Off by default; needs a preparation model in either Basic or AI mode.

## Why

Students ask in everyday words and textbooks answer in their own terms. "What happens when memory
gets too full?" and "buffer pool eviction" share no words, so the passage that answers never reaches
the window. `keyword_terms_for_query` also drops any query term whose document frequency is zero, so
a casual word that appears nowhere in the collection is discarded before BM25 runs. Attaching the
student's vocabulary to the passage gives those words a document frequency and a place to match.

An earlier attempt added separate bridge passages instead. That failed structurally: a generated
passage competes for a window slot, and the embedder cannot tell the right generated passage from the
wrong one (it was admitted on 120 of 128 questions and was off topic on 92 of them), so it displaced
book text about as often as it helped. Writing into the passage's own search terms removes both
problems: nothing is added to the index and nothing is displaced.

## Measured

Generated BuzzDB textbook, nomic-embed-text, K=4 hybrid, gemma4:e4b writing the sentences from the
passage text alone. 56 casual student phrasings covering 14 concepts, scored strictly (one retrieved
chunk must carry at least two of the concept's answer keywords), plus 28 textbook-worded controls.
Every index rebuilt from empty.

| passages given terms | retrieval (56) | won / lost | controls (28) |
| --- | --- | --- | --- |
| none | 40 | | 28 |
| every prose passage, one sentence | 45 | +5 / 0 | 28 |
| the 13 passages that answer the test concepts | 46 | +6 / 0 | 28 |

Answers with gemma4:e4b reading the same windows: 41 of 56 correct without terms, 47 with them on
those 13 passages (+6 / 0). Four of the six come from windows that went from uncovered to covered.

On a broader set (150 passages sampled across the book, one casual question each written by a
different model) giving every prose passage one sentence raised passage recall from 98 to 112 of 150.

Costs, measured through the engine: about 3 seconds per passage once, cached by passage text so an
unchanged document makes no model call on a later run; index time unchanged (62 to 75 s for this
book either way); search latency unchanged; the answer window is unchanged, because the sentence
stays out of the passage text.

## Limits

- The gain is retrieval. A reader-side benefit was not established: of the six answer-level wins only
  four came from newly covered windows.
- It helps a question that shares words with the sentence. "Protecting an in-use page from
  replacement" still missed a passage whose sentence said "stops the system from kicking it out".
- One sentence everywhere is the safe setting. Two sentences per passage reached further on the
  broad set (120 of 150) but was the only configuration that lost human questions (3 of 56).
- A second book (an Operating Systems text) gained only 64 to 66 of 72: its baseline was already 89%,
  it has no chunk markers so sentences landed at section level, and the added text was 16% of a small
  book, the dose where losses appear.
- Sentences are generated, so they can be wrong. The preparation report shows each one under its
  passage, and `chunk_text` is unchanged, so a bad sentence degrades search for that passage but
  cannot put invented text in front of a student.
