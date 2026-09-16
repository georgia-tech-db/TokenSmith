# First-run setup and model workflow

Status: revised plan, not implemented. Audited September 13, 2026 after checkpoint
`ea4904f` (cloud setup, document preparation, and UI refinements).

September 14 decision: Basic is now the implemented default for new collections. AI
remains an explicit experimental option and uses numbered source-line boundaries.
The broader first-run redesign below is still planned, not implemented.

## Decision

Make first run two student-facing tasks: **Choose an answer model**, then **Prepare a
collection**. Present **On this device - Ollama** and **Cloud - API key** equally for
the first task. These choices describe where answers run, not a global privacy mode
for every operation in TokenSmith.

Remove the standalone Models item from primary navigation only after its capabilities
have working replacements. Primary navigation becomes Chat, Library, Settings.
Use the chat picker for choosing/adding answer models, Library for collection search
and preparation choices, and Settings > Models & connections for ongoing management.
Reuse the existing cloud connection service and dialog; do not implement them again.

Do not delete, move, rename, or redownload installed models to test this work. Never
reset the user's TokenSmith profile, collections, credentials, or conversations.

## What was actually checked

The production renderer was opened through an isolated browser fixture with an
in-memory bridge. Ollama availability, installed models, and cloud connection results
were simulated. No real credentials, model calls, downloads, indexing, or provider
requests were used. This verifies UI behavior, not live service compatibility.

| Scenario | Observed behavior | Required change |
| --- | --- | --- |
| Empty TokenSmith state; no Ollama | Setup leads with Install Ollama, Download models, Add PDFs. Cloud is below the steps. The disabled composer says Download Llama 3. | Equal answer-model choices; provider-neutral readiness and composer text. |
| Empty state; Gemma and Nomic already reported by Ollama | Setup still offers Download Nomic and Download Llama 3. Neither installed model appears in the chat picker. | Discover and offer existing compatible models before any download. |
| Existing-model reuse | Models > Explore offers Use Chat Model for Gemma and Use Embedder Model for Nomic. Selecting Gemma works but leaves the user on Models. | Move the reuse action into setup and return to the originating task. |
| New cloud connection | Provider, key, model selection, and return to chat work through the simulated service. The remaining guide still says Install Ollama and Start Ollama first. | Mark answer setup complete and explain only the outstanding collection requirement. |
| Cloud connected; add first folder | Add and prepare is available. Submission then reports Add an embedding model in Models first. | Show the missing search-model action before submission, directly in Library. |
| AI preparation with cloud chat selected | The cloud chat model becomes the preparation model by default. The document-text disclosure is hidden under Customize. | Show the preparation destination and disclosure beside the final action. |
| Chat draft followed by Library and back | The unsent draft is lost because ChatScreen is unmounted. | Preserve drafts across setup/navigation; a modal alone does not fix every route. |
| First-run layout at a 1280 x 720 preview | Empty conversation/source sidebars squeeze setup; action buttons overlap text, and the guide is scrolled down. | Use the available workspace for first run, stack controls by container width, and start at the top. |

Source inspection also confirms:

- `src/shared/ollama.ts` still recommends `llama3`; this is not just stale display text.
- Cloud credentials now have protected persistence, explicit session-only mode,
  connection reuse, cancellation, and a generic selected-model verification request.
  Credential presence after restart is not proof of current provider access or quota.
- `CloudGeneratorService` intentionally supports generators only. Legacy remote
  embedding setup does not yet share that credential lifecycle.
- Basic preparation is now the default and needs an embedder but no generation call.
  Opt-in AI preparation needs a generation model plus an embedder. Neither means
  keyword-only search.
- Collection updates keep the old published index during replacement. The UI must not
  equate prepared/cached passages with newly published searchable passages.

## First-run journey

### 1. Choose an answer model

Show a compact setup surface in Chat, without empty history/source sidebars consuming
most of its width. No marketing page, compulsory account, or full-screen tutorial.
Keep the task title, both execution choices, and their primary actions visible.

**On this device - Ollama**

- Check Ollama and list installed compatible chat models. Discovery is read-only and
  must not register, select, pull, delete, or start inference with a model implicitly.
- Offer Use this model for an installed choice. Keep engine/endpoint/model identity
  and role intact; do not create duplicates by treating tag aliases as new downloads.
- If Ollama is installed but stopped, offer Start/Open and Recheck. If unavailable,
  offer its existing installation link. Never make cloud connection wait for this.
- Only offer a download for a missing model after the user chooses it. Keep progress,
  pause, resume, failure/retry, and cancellation in this flow and in Settings.
- Align the suggested new-install chat model with the already agreed Gemma E4B
  choice (`gemma4:e4b`), through one shared recommendation definition. Derive visible
  labels from that definition. This must not switch existing users off Llama or any
  other selected model, and does not authorize a download during this planning task.
- Use capability/role information where available. Unknown capability is not proof
  that a model can answer questions or embed documents. Validate the operation before
  reporting the model usable; keep that test explicit and free of student documents.

**Cloud - API key**

- Open the existing provider > key > available models > verification dialog directly.
  First-use wording should say Choose a cloud model, not Bring another model.
- Retain billing, secure/session-only storage, manual model entry, cancellation, and
  actionable provider errors. Do not automatically select the first returned model.
- Explain the actual chat uses of the provider: question rewriting, answers, and
  enabled suggested questions, including relevant excerpts/history as applicable.
- Commit selection only after verification succeeds. Preserve the previous working
  selection on failure; keep disconnected saved models visible for repair.
- When complete, show the chosen answer model and the next outstanding collection
  task. Do not prompt a cloud user to download a local chat model.

### 2. Prepare a collection

Allow a student to choose a folder and name before setup is complete. Preserve those
choices while configuring missing dependencies. Before Add and prepare, show:

| Setting | Normal first-run behavior |
| --- | --- |
| Search model | Reuse the configured compatible embedder, or offer setup inline. Label its execution location. Nomic remains the existing local default; installing it is an explicit action. |
| Preparation | Default to Basic. Offer AI explicitly, labeled experimental, using the chosen answer model unless the student selects another one. Preserve existing collection choices. |
| Data destination | Show where preparation and embedding run separately. Cloud AI preparation sends document text, not just the short excerpts used for an answer. |
| Final action | Ready only when the chosen preparation mode and search model have their required dependencies. Explain any blocker with an actionable setup control. |

Do not ask students to configure three distinct models by default: the answer model
can also do AI preparation. Search embeddings remain a separate role. Basic preparation
must not falsely require a chat model; local/cloud chat must not silently reconfigure
the embedder for an existing collection.

For cloud answers with local search, state the remaining Ollama/search-model requirement
plainly. Preserve the existing advanced remote-search option, but do not advertise it
as a finished no-Ollama first-run path: its secure credential/reconnect support is not
equivalent to cloud chat. Promoting cloud search requires a separately reviewed,
role-aware extension of credential resolution, embedding validation, and restart tests.
Never broaden the generator-only credential resolver by simply removing its role guard.

Show the selected preparation provider and potential usage before starting a job,
outside collapsed settings. Freeze the job's selected models/settings when submitted;
changing the chat picker later must not change an in-flight preparation job. Failed
preparation must not silently fall back to another model or to Basic mode.

Ready for chat means content has actually been published into the searchable index.
Distinguish preparing, paused, failed, and using a previous index. Keep existing
publication semantics; this UI refactor does not redesign incremental indexing.

## Remove the separate Models pane safely

| Existing capability | Destination |
| --- | --- |
| Choose a chat model; reconnect saved cloud model | Chat model picker |
| Add/reuse/download an Ollama chat model | Shared local-model setup sheet opened from Chat or Settings |
| Connect cloud chat; add another model to a connection | Existing cloud setup dialog opened from Chat or Settings |
| Choose/add a collection embedder | Library collection setup, with shared model setup for the search role |
| Choose AI preparation model and instructions | Library collection customization |
| Manage downloads, endpoints, credentials, model removal | Settings > Models & connections |
| Per-model generation parameters and prompts | Existing settings, clearly grouped under answer-model settings |
| Legacy remote embedding configuration | Advanced search-model settings, not silently removed |

Extract the existing management logic rather than copy-pasting it into several screens.
Mount setup dialogs at App level with a small origin/role contract. On completion or
Cancel, return focus to the triggering task. Keep model mutations independent of
navigation: `upsertModel` must no longer default to `activeScreen: 'models'`.

Keep the whole app state out of a new onboarding state machine. Derive readiness from
the selected usable models, collection publication state, and preparation mode. Use
local component state only for a sheet's current step and unsaved non-secret choices.
Treat stopped Ollama, provider offline/quota errors, missing credentials, and absent
models as distinct recoverable conditions. Do not delete configuration on probe failure.

Preserve unsent chat drafts by conversation ID and collection form drafts across
necessary navigation. Never persist raw unsubmitted API keys in general app state.
Migrate a saved Models screen to Settings > Models & connections without changing
model IDs, selections, collection bindings, credentials, or indexed content.

The layout should follow its actual container width. Hide empty sidebars in first run;
restore ordinary conversation/source controls once useful. Setup is not a chat turn:
it must not inherit automatic scroll-to-bottom behavior. Removing the last usable
model should reveal a repair action, not force a destructive first-run reset.

## Implementation order

1. Extract shared local-model setup and model-management controls. Keep existing
   download handlers and cloud service intact. Add direct picker and Library entry
   points; preserve task drafts and return focus.
2. Replace the Ollama-only guide with the two-task setup flow. Add installed-model
   reuse, neutral readiness, one recommendation source, and responsive empty state.
3. Add Library preflight, visible model/data destinations, and accurate publication
   status. Keep AI and Basic preparation requirements distinct.
4. Move remaining controls to Settings and migrate the old route. Remove the primary
   Models item and duplicated/dead first-run handlers only after all entry points work.
5. Run the safe test matrix below before packaging. Treat improvements to AI boundary
   quality and a full cloud-search lifecycle as separately scoped changes, not implicit
   prerequisites for fixing cloud answer-model discoverability.

Likely ownership: `App.tsx`, `ChatModelPicker.tsx`, shared local setup/management
components, `CloudGeneratorDialog.tsx`, `LibraryWorkspace.tsx`, `shared/ollama.ts`,
and saved-state normalization. Generation prompts, retrieval ranking, and model files
are outside this refactor.

## Verification without deleting models

Automated UI fixtures must use production components with an in-memory bridge. Reset
only their fixture state. Record calls and fail tests on unexpected pulls, deletes,
provider requests, or real-profile access. No models need downloading for CI.

| Test | Required assertion |
| --- | --- |
| Fresh state, no Ollama | Both answer choices visible; cloud flow opens without waiting for Ollama. |
| Fresh state, installed Gemma/Nomic | Both can be reused; zero pull/delete calls; no mandatory Llama download. |
| Ollama installed but stopped | Start/Recheck offered; saved choices preserved after failure. |
| Cloud answer ready, no embedder | Answer step complete; Library exposes search setup before import. |
| Embedder ready, no chat model | Basic preparation allowed; AI preparation explains its missing generation model. |
| Local chat and local search ready | No redundant setup/download; collection import remains available. |
| Cloud preparation selected | Document-text destination visible before submission, without expanding Customize. |
| Cancel/failure/return | Chat draft, collection path/name/instructions, selection, and focus preserved. |
| Existing indexed collection | Chat-model switch never changes its embedder or starts reindexing. |
| Download pause/resume, refresh, restart | State remains repairable and no duplicate model is created. |
| Legacy Models route | Opens new settings destination with IDs and references intact. |
| Narrow and desktop layouts | No overlapping buttons/text; both choices readable; initial scroll at top. |

Run existing TypeScript/Python suites plus focused readiness/migration and UI contract
checks. Do not report mocked connection or preparation tests as real LLM accuracy.

For an optional native walkthrough, use a dedicated temporary user-data directory and
reuse the existing Ollama store read-only. A fresh TokenSmith profile does not isolate
Ollama's shared model store: explicitly block pull/delete routes or use a mock Ollama
endpoint. Simulate a missing runtime/model in the fixture, never by modifying the real
installation. The current user profile and model directory must remain untouched.

Baseline checkpoint verification: 121 TypeScript tests, 91 Python tests, and the
production build passed against the exact staged tree. The new first-run flow itself
has not been implemented or acceptance-tested yet.
