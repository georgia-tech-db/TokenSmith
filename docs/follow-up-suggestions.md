# Suggested follow-up questions: grounding and cost

## Problem

Follow-up suggestions were generated from the previous answer alone. Whatever the answer added on
its own became the next question. One logged chain on a Chapter 5 collection: the app suggested
"Why does it use LRU instead of FIFO?" (FIFO is not in that chapter), answered it from model
knowledge, then suggested five consecutive questions about a recency-versus-frequency distinction
the material never makes, ending with invented C++ presented as an example.

## What changed

1. **The suggester sees the retrieved excerpts.** `followUpSuggestionMessages` now includes the
   same packed source block the starter questions already use, and the default follow-up prompt
   asks for questions about ideas that appear in the excerpts and refuses to build on anything the
   question or answer assumes that the excerpts do not state. The previous default is recognised as
   legacy so existing users are upgraded; custom prompts are kept.
2. **A local answer model writes its own follow-ups.** It is already loaded and its prompt prefix
   is cached, and a second model evicts it on smaller machines (measured: Gemma 4 E4B reloads for
   7 to 9 s after a 3B follow-up call on a 16 GB laptop). A cloud answer gets follow-ups from the
   smallest installed local chat model, never from the cloud; with no local model installed it gets
   none.
3. **Two more filter rules** in the existing `filterSuggestedQuestions`: a candidate that
   near-duplicates one already kept is dropped, and a candidate that names a code identifier absent
   from both the question and the answer is dropped (the prompt's own rule, enforced).
4. **The follow-up prompt opens like the answer prompt** (same system message, then the same
   excerpt block), so Ollama can reuse the answer prompt's cached prefix when the two models are the
   same.

## Measured

Same 25 turns each time (13 from the logged cascade conversation, 12 ordinary turns), run through
the engine's own functions against Ollama on a 16 GB Apple laptop, medians per follow-up call.
"Bad" is hand-labelled by one rater: off the material, built on a claim the answer invented, or
inverted.

| setup | bad (cascade, ordinary) | shown of 52 / 48 | prompt tokens | generated tokens | seconds |
|---|---|---|---|---|---|
| shipped 0.1.14 (llama3, no excerpts) | 15/52, 3/47 | 52, 47 | 452 | 63 | 5.8 |
| new prompt + excerpts, gemma4:e4b writes its own | 1/44, 1/42 | 44, 42 | 2399 | 54 | 4.6 |
| new prompt + excerpts, llama3 writes its own | 14/49, 4/46 | 49, 46 | 1868 | 70 | 6.5 |
| new prompt + excerpts on llama3.2:3b (second model) | 4/39, 2/31 | 39, 31 | 1883 | 53 | 5.5 |

Findings that shaped the choice:

- With the recommended model (Gemma 4 E4B) the prompt and excerpts are enough: it declines the
  invented premises on its own (no FIFO, no invented optimization, no MRU code) and still fills the
  list. llama3 with the same prompt inherits the invented claims almost as often as before; for a
  llama3 user the 3B model would give better follow-ups, but on a 16 GB machine a second model
  evicts a Gemma answer model, so the answer model writes its own.
- Reading the excerpts is half or more of each call (prefill), so the prompt size, not the output,
  is the cost. Clipping excerpts to 1600 characters per source barely reduced tokens (sources are
  already that short); 800 characters halved the shown suggestions.
- Asking for six candidates to show four did not increase the number shown and cost about 30
  generated tokens a turn, so the request asks for exactly the shown count.
- Two resident models: llama3 (4.7 GB) and the 3B fit together in 16 GB (no reload, 181 ms), but
  the 3B's prefill is five times slower beside llama3 (0.7 s to 3.7 s). Gemma 4 E4B (9 GB) and the
  3B do not fit: the 3B call evicts Gemma and the next answer reloads it (7.3 to 8.9 s measured).

Not included, and why: an answer-to-source word-overlap gate (fired on none of 12 unseen turns and
shares the weakness discussed on PR #140), any output-token cap (real responses are about 90 tokens
against a 384 cap), and any change to the answer path.
