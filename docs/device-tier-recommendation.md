# Model recommendations for course study

Start with Gemma 4 26B A4B when it meets the device policy, prioritizing its measured balance of answer coverage and wait time for interactive study. Otherwise offer 12B, then E4B, based on their requirements. This is a fixed preference among evaluated options, not a rule to pick the largest model available. Eligible alternatives appear separately in Models; existing installed models and user selections remain available. When no option qualifies, setup suggests cloud as an option, not a prohibition on manual local use. Connecting a cloud provider remains explicit.

## Evidence and limits

The September 27, 2026 BuzzDB experiment replayed 18 questions from six related families on a 32 GB M1 Max, using frozen Nomic passages and original E4B conversation history. Q4_K_M packages, 8,192-token context, 1,536-token answer allowance, thinking off, and matching API sampling settings were used. Each final generation had a one-token preflight. No new retrieval or rewriting was timed.

| Package | Coverage /72 | Answers with material errors /18 | Median preflight + answer |
| --- | --- | --- | --- |
| Gemma 4 E4B | 55 | 9 | 34.2 s |
| Gemma 4 12B | 67.5 | 2 | 126.8 s |
| Gemma 4 26B A4B | 66.5 | 5 | 43.1 s |
| Gemma 4 31B | 67 | 4 | 233.4 s |

These are non-blind, single-sample manual results on previously known questions, not general accuracy estimates, learning outcomes, or speed predictions for another machine. Coverage does not cancel false explanations. Runs were sequential, not interleaved. The 26B package includes speculative decoding, so this is not an isolated architecture comparison.

26B is preferred when eligible because its measured answer stage was about three times faster than 12B, with similar coverage. It had more error-containing answers (5 versus 2); this choice is a product trade-off, not a claim of higher correctness or proven learning gains. 12B remains a lower-memory alternative with fewer observed errors, and E4B is the lighter fallback. 26B has more memory demand: runtime logs recorded 16,147 MiB of main GPU weights plus 425 MiB of draft weights, before other allocations. System swap grew from about 8.4 to 15.1 GiB in that run; causation cannot be assigned entirely to the model. Ollama's approximately 1 GB residency report for this package was incomplete. Do not treat four active billion parameters as a four-billion-parameter memory footprint. 31B is not in the recommended catalog because the trial did not justify its latency; manual installation remains possible.

## Conservative eligibility policy

| Option | Host RAM | CPU-only | Unified RAM | Dedicated VRAM available and total | Free disk |
| --- | --- | --- | --- | --- | --- |
| E4B fallback | 16 GiB | 24 GiB, 8 threads | 16 GiB | 10 GiB | 14 GiB |
| 12B fallback | 24 GiB | Not recommended | 24 GiB | 12 GiB | 14 GiB |
| 26B A4B preferred | 32 GiB | Not recommended | 32 GiB | 24 GiB | 25 GiB |

Thresholds are policy estimates, not measured minimum requirements. They assume context, runtime overhead and Nomic retrieval, but cannot guarantee headroom for all other applications. Download size is not runtime memory. The 32 GiB unified threshold admits the benchmarked Mac despite observed paging; the policy records that pressure below 48 GiB for diagnostics. The 43.1-second median includes preflight and generation, excludes retrieval and rewriting, and is neither time to first token nor a latency guarantee. More memory offers headroom but has not been benchmarked here.

Shared graphics memory is never added to host memory. Unknown GPU support cannot qualify as accelerated execution. Known busy VRAM disqualifies that GPU path; CPU fallback still needs its own thresholds. Unknown free VRAM is disclosed. OS/architecture checks and detector limitations remain separate from performance estimates. Total RAM is not an available-memory or pressure measurement, particularly on macOS; rechecking cannot guarantee freedom from swapping. Disk checks use the app's storage volume, which may differ from a customized Ollama model directory.

First-run download buttons use the detected recommendation alongside Nomic. Chat downloads wait for device detection. If detection fails, the generic Ollama fallback remains E4B; no existing model is replaced. The interface shows the recommended model, alternative names, and collapsible device details. Benchmark discussion and policy caveats stay in this document.

## Runtime and existing users

Fresh settings use an 8,192-token context and a 1,536-token answer cap, matching the recommendation assumptions. Saved settings and per-model overrides are preserved. This change does not add generation preflights, change thinking behavior, or reproduce the benchmark runner inside the app. Recommendations themselves make no model calls.

## Next validation

Benchmark fresh multi-turn questions and full retrieval on representative CPU, Apple Silicon, and dedicated-GPU machines, with normal student apps open. Measure time to first token, full-answer time, peak memory, swapping, and material errors. Revisit the 26B default if full-session latency or errors outweigh the observed answer-stage speed advantage.

## Catalog provenance

Ollama's [Gemma 4 tags](https://ollama.com/library/gemma4/tags), checked September 28, 2026, list the catalog aliases as Q4_K_M artifacts. Approximate downloads: E4B 9.6 GB, 12B 7.6 GB, 26B 19 GB. Aliases can change; revalidate the package before updating benchmark claims. Retained experiment artifacts live under `tmp/deep-course-eval/mid-gemma/` and `tmp/deep-course-eval/larger-gemma/` in the benchmark workspace; they are not shipped with the app.
