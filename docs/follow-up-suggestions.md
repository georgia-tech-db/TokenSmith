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
2. **Follow-ups run on the smallest installed local chat model.** They are a side task, so the
   engine picks the smallest non-embedding model in Ollama's list. A cloud answer gets local
   follow-ups; with no local model installed it gets none. The answer model is used only when it is
   the sole local model.
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
| new prompt + excerpts on llama3, cached prefix | 14/49, 4/46 | 49, 46 | 1868 | 70 | 6.5 |
| new prompt + excerpts on llama3.2:3b | 4/39, 2/31 | 39, 31 | 1883 | 53 | 5.5 |

Findings that shaped the choice:

- The grounding gain comes from the small model, not the prompt: llama3 with the new prompt
  inherits the invented claims almost as often as before. The 3B model also returns an empty list
  on the poisoned turns (the fake MRU code, the inverted "LRU helps scans" premise), which is why
  it shows fewer suggestions.
- Reading the excerpts is half or more of each call (prefill), so the prompt size, not the output,
  is the cost. Clipping excerpts to 1600 characters per source barely reduced tokens (sources are
  already that short); 800 characters halved the shown suggestions.
- Asking for six candidates to show four did not increase the number shown and cost about 30
  generated tokens a turn, so the request asks for exactly the shown count.
- With llama3 resident beside it the 3B model's prefill is five times slower than alone (0.7 s to
  3.7 s); both fit in 16 GB and the answer model does not reload afterwards (181 ms). On a smaller
  laptop a second resident model is the open risk.
- The 3B model asked for the answer model's cached prefix cannot help, since they are different
  models; the cache rule matters only when the answer model is the only local one.

Not included, and why: an answer-to-source word-overlap gate (fired on none of 12 unseen turns and
shares the weakness discussed on PR #140), any output-token cap (real responses are about 90 tokens
against a 384 cap), and any change to the answer path.
