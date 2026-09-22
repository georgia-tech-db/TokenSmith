# Bridge chunks

Opt-in corpus augmentation for the query-side vocabulary gap. **Off by default**, and the
measurements below are the reason it should stay off until someone repeats them on another book.

## The problem

Students ask in everyday words; textbooks answer in their own terms. "What happens when memory
gets too full?" and "buffer pool eviction" describe the same thing but share no words, so the
embedding never gets close enough and the right passage never reaches the model.

## What a bridge chunk is

A short passage in student wording that pairs a casual question with the book's actual mechanism:

    Question: how does it figure out which page to get rid of?
    When the pool is full, evict() removes the least recently used page from the lruList,
    writing it to disk first ...

Indexed next to the book, it is what a casual question matches, and it carries the answer. The
content is what matters: a bridge that only names the term barely helps; one that carries the
mechanism moves answers.

## How it works

After a material is embedded, in both the Basic and the AI-prepared indexing paths, and only when
the caller sets `generateBridges`:

1. **Inventory** group chunks into concepts, by source unit when the material has them, else by
   section header. One representative (longest) chunk per concept.
2. **Audit** the generator writes a couple of casual phrasings per concept; each is embedded with
   the material's own embedder and checked for whether the concept's chunk lands in the top-K
   window. Concepts whose phrasings miss are the gap concepts.
3. **Budget** gap concepts are ranked by how badly they miss, and only the top 5% of the chunk
   count (floor 8) is bridged, so augmentation cannot crowd real passages out of the window.
4. **Bridge** for each kept concept the generator writes a plain-words explanation that must keep
   the chunk's distinctive terms and must not talk *about* the passage; an answer that fails that
   check twice is dropped rather than indexed. One chunk is emitted per missing phrasing.

Exercise sections ("Check Your Understanding" and the like) are never bridged: a quiz question
rewritten as a student question would retrieve the quiz, not the answer.

Bridges are stored with `chunk_kind = 'bridge'` and no `parent_id`, so source-unit expansion
passes them through untouched.

## Retrieval rules

Bridges must never make retrieval worse for the book itself, so they are kept out of the ranking
that selects book passages:

1. They are excluded from the keyword (BM25) arm in `_active_chunks_filter`, and held out of rank
   fusion entirely in `search_library`. The vector arm over-fetches by the number of bridges in
   the collection, so removing them still leaves a full book candidate list.
2. At most one bridge is then admitted, chosen by **vector similarity only** (never by shared
   everyday words, which is how a wrong-topic bridge would otherwise win), and only above
   `BRIDGE_MIN_SCORE`. It takes the first slot when it matches the question better than the best
   book chunk, otherwise the last. The window size does not change.

An earlier version mixed bridges into hybrid fusion. The keyword arm then pulled in bridges for
the wrong concept, and answer accuracy on casual phrasings fell to 16/28 against 20/28 without
bridges. That is why the exclusion above is structural.

### Verified, and where it does not hold

Indexing one chapter twice, with and without bridges, and comparing the book chunks each query
returns (5 casual questions, 111 chunks against 119):

| mode | book window identical | bridge admitted |
|---|---|---|
| vector | 5/5 | 5/5 |
| **hybrid** (default) | **5/5** | **5/5** |
| keyword | 4/5 | 0/5 |

Keyword-only mode is the honest exception, for two reasons. Bridges are written into `chunks_fts`
by the insert trigger, so they join the BM25 corpus statistics (document frequency, average
document length) even though `_active_chunks_filter` keeps them out of the results. On one of five
questions that moved the fourth book slot. And because admission is decided on vector similarity,
keyword-only search can never admit a bridge, so there the cost is paid with no upside.

Keeping bridges out of the FTS index entirely would fix both, but it means migrating the three
`chunks_fts` triggers, which is more surgery than an off-by-default experiment justifies.

## The measured tradeoff

On a generated database textbook, using the app's own retrieval (nomic embeddings, top 4):

| | casual phrasings answered |
|---|---|
| plain retrieval | 39% |
| with generated bridges | 61% |
| with hand-written bridges | 68% |

Held-out phrasings the generator never saw. That is the case **for** bridges.

The case against, measured on the same engine:

- **The gap is mostly already closed.** On 0.1.13 the same casual set scored 25/28, and a failure
  decomposition attributed only **3** failures to retrieval against **4** to the reader. Bridges
  can only address the first group.
- **They cost real passages.** Admitting a bridge evicts a book chunk from a 4-slot window. On the
  0.1.13 set this moved casual answers 25 -> 23 and in-vocabulary answers 28 -> 26.
- **An extra slot is worse, not better.** Adding the bridge as a fifth source instead of replacing
  a book chunk scored 20/28 against 22/28: the extra context distracts a small reader.

The honest summary is that the mechanism works and is measurable, the window cost is real, and on
a current build the problem it solves is no longer the dominant failure. It is worth keeping as an
opt-in experiment and worth re-measuring on a second book before anyone considers defaulting it on.

## Costs and limits

- Upload time grows by the generation calls: one phrasing call per concept plus one explanation
  call per bridged concept. On a 411-section book that measured roughly 17 minutes with a 3B
  generator and 31 minutes with an 8B one on a laptop.
- Index size grows by up to `N_PHRASINGS` chunks per bridged concept, not by the budget itself.
  The budget caps *concepts* at 5% of the chunk count (floor 8); each can emit up to two bridges.
  Measured on one chapter: 111 chunks to 119, about 7%.
- The generator is the weak link, not the retrieval rule. Without the commentary filter, 6 of 11
  bridges on this chapter opened with "This passage describes a mechanism..."; with it, 8 of 8
  pass, but they are 8B-model prose: one phrasing ("what are the pros and cons of...") is a weak
  question, and one answer explains hit/miss where eviction was asked for. Read a few before
  trusting a bank.
- Only Ollama generators are supported; with any other generator the step is skipped.
- No resume: an interrupted upload regenerates the audit from scratch.
- Bridges cannot help when retrieval already succeeded and the reader ignored the passage, which
  is a distinct and currently larger failure mode.
