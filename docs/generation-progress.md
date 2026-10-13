# Generation progress

A single bottom status strip is shared by Chat, Library, Models, and Settings. It shows the current operation, an estimated time range, and Stop. Its label opens the associated conversation. Completion appears briefly; the strip retains its space when idle. Narrow windows use two rows, and reduced-motion preferences disable animation.

Answers are delivered through a request-scoped IPC notification as soon as generation and source ordering finish. The existing follow-up call then runs; its result updates the same message. Starting another question cancels those background follow-ups while preserving the completed answer. No additional model calls are introduced. Stop forwards cancellation to the active Ollama or remote completion, including question rewriting and Gemma prompt preflight. An in-flight library search may finish, but cancellation prevents it from launching the next inference. Automatic starter suggestions retain their existing cancellation on leaving Chat.

## Estimates

The renderer measures the whole wait from submission through answer availability, including rewriting, retrieval, and preflight. Follow-up suggestions are timed separately. Starter timing begins when its debounced request is dispatched. Quiz and simpler-explanation operations also use the strip.

Initial estimates are deliberately broad. The Gemma answer priors use the earlier local trials as rough starting points (40 seconds for E4B, 45 for 26B, 125 for 12B), adjusted for output limit, thinking, conversational context and absence of detected GPU support. Other models use generic priors. These are not promises for a particular machine; memory pressure, model loading, input length and output length remain variable. Remote inference is learned per endpoint; local hardware detection cannot predict its speed.

Successful completions update a recent weighted duration estimate. After one observation, 80% of the estimate comes from the observed duration; with more observations, recent samples replace the prior. Configuration keys separate model, endpoint, operation, context/output limits, thinking and compute settings. Input-size buckets, conversation context, question count and explanation depth refine the match. If that exact bucket is new, the same model/runtime/task family supplies a broad fallback. The last twelve matching samples are weighted toward recent runs. Slow completed runs are retained and widen uncertainty; failed or canceled phases do not train the estimator. A successful answer still counts if its subsequent suggestions are canceled.

Up to 200 duration records from the last 90 days are saved in installation-local storage under `tokensmith-generation-timings-v1`. Only opaque configuration keys, durations and timestamps are saved, not prompts, source text or credentials. Storage failure falls back to in-memory learning. There is no calibration inference, telemetry or settings switch.

The progress bar estimates elapsed time within the expected range; it does not measure the fraction of generated text. Once a request outlasts the upper estimate, the bar becomes indeterminate and reports elapsed time. Only actual completion shows 100%.

Tests cover learning, persistence, isolation between configurations, uncertainty, overdue behavior, early answers for both providers, inference-call counts, and cancellation during response-body reads. The disposable full-app UI fixture is documented in `tests/ui/README.md`. Actual prediction accuracy should be evaluated on real usage; one completed run is useful calibration, not a speed guarantee.

## Source reading while waiting

With Show Sources enabled, a normal answer opens its first retrieved PDF or Markdown/text source in the chat area as soon as retrieval finishes. The existing page/passage positioning is reused. The source load runs alongside generation and introduces no model call. Simpler explanations can similarly preview their saved first source; quiz question generation does not reveal its source automatically.

The automatic reader closes at the first answer-ready event, before follow-up suggestions finish. Stop, failure, leaving Chat or switching conversations also dismiss it. A request-owned preview controller prevents late document loads from reopening after completion or dismissal. Manually opening a source takes control away from the automatic preview, so answer arrival never closes the student's chosen document. The automatic reader is non-modal and confined to Chat, keeping navigation and the bottom progress controls accessible.

## Reader navigation

Markdown readers have a collapsible Contents panel with links to parsed headings. Heading levels determine indentation; duplicate headings receive distinct targets, and code fences do not create entries. The expanded/collapsed preference is saved locally. Clicking a heading scrolls and focuses that section; Back to chunk returns to the cited passage. In narrow windows, Contents sits above the document.

Previous/Next source controls appear in both Markdown and PDF readers. In Chat they follow the answer's full source list, retaining separate cited passages from the same file. In Library they follow the selected collection's files, preserving the initially clicked passage and opening other files at their beginning. Mixed Markdown, text, and PDF collections switch readers automatically; PDF page controls remain separate. The current document stays visible during a source load or a load failure, and closing the reader invalidates late loads. Navigating an automatically opened source takes manual ownership, so answer arrival leaves the chosen reader open. Navigation and Contents require no model calls.
