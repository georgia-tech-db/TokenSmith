# Cloud model setup: student experience plan

Status: implemented in the working tree, September 13, 2026. The scope is cloud **generation** models. Nomic, embedding selection, and existing indexes remain unchanged. The design below records the original proposal; the implementation and verification notes here describe what shipped in this change.

## Implementation and verification

- Chat now has a searchable picker grouped into “On this device” and “Cloud,” with a direct “Connect a cloud model” action. Models management uses the same connection sheet.
- The sheet walks through provider → API key → discovered chat model → minimal live validation → return to chat. Known providers require no endpoint entry. The student explicitly chooses a model; there is no arbitrary default based on list order.
- Connections can be remembered with Electron's OS-backed encryption or kept for one session. Keys are held in the main process after submission and are absent from renderer responses and ordinary app-state JSON. Linux plaintext storage is rejected. Reconnecting a saved connection repairs its associated generators.
- Disconnected generators stay visible with a reconnect action. Setup failure and cancellation preserve the previous model, draft, and sources. Network failures offer retry; manual entry is offered only for catalog/model failures. Gemini discovery excludes ordinary embedding, image, video, music, agent, and live-only entries; selected-model validation remains necessary because catalog membership does not guarantee availability.
- Build and type checks passed, along with all 121 TypeScript unit tests, including 21 cloud connection tests. A new indexing shutdown listener needed an optional Electron-app guard so Node-based unit tests could import the shared engine helpers; behavior within Electron is unchanged.
- Walked the production UI using both a deterministic browser fixture and the rebuilt native Electron app. Verified the draft, prior answer, selected source collection, and composer focus survive cloud setup.
- With the user's explicitly provided Gemini test key, real native discovery succeeded, `gemini-3.1-flash-lite` passed the generic connection probe, and one B+ tree range-query question returned an answer with four retrieved passages in approximately 2.5 seconds. The sample book used the existing 1,574-chunk Nomic index. The key was used for this session only and did not enter the credential file or saved app state.
- Gemini's catalog included `gemini-2.5-flash-lite`, but that model returned HTTP 404. Another listed model worked. A later check in the normal app profile reproduced a discovery timeout; retry recovered it. The underlying network cause is not established. Errors now distinguish timeouts, certificate failures, malformed catalogs, and unavailable models without exposing provider bodies or keys.
- Cloud discovery, validation, and generation now use Electron's desktop network transport, with browser cookies omitted. This respects system proxy configuration, unlike the previous Node transport. The local Nomic embedding path is unchanged. Both network transports reached Gemini in a separate read-only probe; the observed timeout was intermittent rather than proof of a specific proxy failure. See [Electron networking](https://www.electronjs.org/docs/latest/api/net).
- Secure persistence was exercised separately using a fake key against a local provider, producing only an encrypted credential record with file mode 0600. A fresh Electron process decrypted that record and restored its ready state; the session-only Gemini model correctly required reconnect. Nomic remained selected for embeddings.

The installed `/Applications/TokenSmith.app` is a separate older build. Initial live checks used an isolated sample-data profile. The final desktop transport was then verified in the normal TokenSmith profile: discovery, validation, and generated study suggestions succeeded, Gemini was selected for this session, and the Models screen still showed Nomic as the active PDF-search model. The named “TokenSmith Cloud Preview” desktop build is left open for review. Provider billing, Windows/Linux OS-store behavior, and a student usability pilot have not been independently measured in this change.

## Decision

Make connecting a cloud model part of choosing a model in chat. The student should choose a service, connect it once, choose an available chat model, and return to the question they were writing. Keep the Models pane for ongoing management and advanced setup.

Start with the existing supported services: Gemini, Groq, OpenAI, Mistral, and a separate custom endpoint option. Adding new provider protocols is outside this first change.

## What the current app makes difficult

| Observed behavior | Student impact | Proposed change |
| --- | --- | --- |
| Chat picker lists only existing models; its empty state is disabled in the source | No obvious way to get a cloud model from the place where it is needed | Always available picker with “Connect a cloud model…” |
| Models → Add Model opens Ollama; cloud setup is behind “Providers” | Cloud setup is buried behind a local-model workflow | Direct cloud action in chat and Models; label choices “On this device” and “Cloud” |
| Several full provider forms appear simultaneously | Many fields and technical choices before the student has chosen a service | Compact provider list, then one focused connection sheet |
| API Key, Model Type, Model Name, Load Models, and Install all appear together | The student must understand the implementation and guess the sequence | Connect → choose chat model → Use this model; automatic discovery |
| The cloud action is named “Install” | Suggests a download that will not happen | “Connect” and “Use this model” |
| API keys are stored in an in-memory map and removed from saved state | Model access does not survive app restart | Persist provider credentials using protected OS-backed storage |
| A remote model without an inline key becomes `needsRuntime`; installed-list and chat filters exclude it | A connection problem can make a saved model disappear | Keep it visible as “Reconnect required” |
| Non-empty key, endpoint, and model name are sufficient to mark a model ready | Errors can appear only after a student asks a question | Verify selected-model access before committing a ready connection |
| Model setup state belongs to the mounted Models screen | Navigating away can discard the draft | Own connection flow at app level; retain non-secret choices and the in-memory draft during navigation |
| Adding a model selects it but navigates to Models | Setup does not return to the task that prompted it | Restore the same conversation, draft question, sources, and focus |

The key-loss and hidden-model findings are from source inspection, not a live restart experiment with a real API key.

## Patterns to borrow

### Ollama — inspected directly

The chat composer has a searchable model picker containing local and cloud choices. Cloud entries carry cloud icons. Selecting a cloud model while signed out reveals “Cloud models require an Ollama account” and a “Sign In” action immediately above the composer. The student does not first configure a model object in a separate management pane.

Borrow the in-chat entry point and the contextual action needed to make a selected model usable. Keep TokenSmith's existing top-bar picker position initially; relocating it to the composer is an optional layout choice, not necessary for the improvement.

Ollama authenticates access to its own service. This does not imply that TokenSmith can provide the same account sign-in for every independent provider. The first release uses the API-key mechanisms TokenSmith already supports.

### Msty — official documentation, not a locally installed app

[Msty Studio's provider flow](https://docs.msty.ai/studio/managing-models/online-providers) treats credentials as a provider connection, then allows the provider's models to be selected in chat. Its [earlier model selector](https://docs.msty.app/features/model-selector) brings local and remote models together and supports frequently used choices.

Borrow connecting a provider once, using friendly model names, and keeping ready models easy to find. Avoid copying its larger settings hierarchy into the student workflow.

## Proposed student flow

### 1. Discover from chat

The existing model label opens a compact searchable menu:

- Current/ready models, with readable names and an explicit “On this device” or “Cloud” label.
- Saved models needing attention, with “Reconnect” as an actionable state.
- “Connect a cloud model…” as a persistent, visible action.
- “Manage models…” as a secondary action.

With no chat model, show “Choose a model” and the same usable menu. Do not require a local chat download before connecting a cloud model.

### 2. Choose a service

Open a single connection sheet over the current chat. Show compact provider choices, not a grid of complete forms. Use known names: Google Gemini, OpenAI, Groq, and Mistral. Put “Custom connection…” behind an advanced option.

Do not label one service “best” or “free” without maintained evidence. Choose a sensible default chat model only after checking current availability and supported capabilities.

### 3. Connect once

The sheet asks for an API key for the selected service. It includes:

- A direct “Get an API key” link to that provider's existing catalog URL.
- A masked key field and an optional show/hide control.
- “Remember securely on this device” with a session-only alternative.
- One sentence explaining provider billing; for services with a separate consumer chat subscription, clarify that distinction using verified provider-specific help.
- “Connect,” which checks the credential and loads available models. There is no separate “Load Models” task.

Opening a key-help page and returning should preserve the sheet and its draft. Do not log or persist the unsubmitted key in ordinary app state.

### 4. Pick and use a chat model

After connecting, reveal a short list of supported chat models in the same sheet. Preselect a maintained recommendation when one is actually available; provide “All available models” for other choices. Display the exact model identifier as secondary detail when useful. Do not default to whichever model happens to sort first.

“Use this model” performs a minimal, generic capability check, saves the connection, selects the model, and returns to the existing chat. Explain that this small test may incur provider usage. The test never includes the student's messages or PDFs. A successful model-list request alone does not prove that a selected model supports the application's chat endpoint or has usable quota.

Before final use, state plainly: “Chat messages and relevant PDF excerpts are sent to [provider].” For the existing local-search setup, also say “PDF search stays on this device.” Audit automatic suggestions and question rewriting so this disclosure covers all cloud calls triggered by the selection. Do not promise all-local processing when a cloud embedder is active.

Show a brief “Connected · [model]” confirmation in chat. Keep the draft question unchanged and ready to send; never automatically submit it as a connection test. Choosing a chat model must not change the collection's embedder or trigger reindexing.

### 5. Reuse and recover

The provider is connected once. Adding another model from it does not request the key again. A revoked key produces an inline “Reconnect [provider]” action that opens the same sheet with the provider and selected model retained. Repairing the connection repairs all models using that connection.

Existing credential-less remote entries are migrated to visible reconnect states. Preserve model IDs and references where possible so conversations and collection embedding references are not broken.

## Errors that keep the student moving

| Condition | Message/action | State behavior |
| --- | --- | --- |
| Missing key | “Paste your [provider] API key to continue” | Stay on key step |
| Invalid/revoked credential | “This key wasn't accepted. Check it or get a new key.” | Keep provider and model; focus key field |
| Network timeout/offline | “Couldn't reach [provider]. Check your connection and try again.” | Keep choices; Retry; available local-model option |
| Permission/model access error | “This account can't use [model]. Choose another model.” | Keep provider connection and show alternatives |
| Billing/quota exhausted | Provider-specific explanation and billing link | Do not label the key invalid or silently switch providers |
| Temporary rate limit | “Too many requests right now. Try again shortly.” | Honor retry information and keep connection |
| Empty or unavailable model catalog | “Couldn't load model choices.” | Retry; advanced manual model entry; still test actual model access |
| OS credential storage unavailable/locked | “Unlock secure storage or use this session only.” | Never silently save plaintext; never report a persistence success |
| Provider/model retired | “This model is no longer available.” | Keep history and offer a user-selected replacement |

Classify errors by provider response details as well as status code; a 429 may mean rate limiting or exhausted quota. Bound connection and discovery requests with cancellation/timeouts. Ignore stale responses after changing a key, endpoint, provider, or model. Keep the previous working model selected until the replacement is successfully verified and committed.

## Management surface

Keep “Models” in the existing navigation for the first release. Use compact rows grouped under “On this device” and “Cloud connections,” with readable names, connection status, and a clear “Connect cloud model” action. Put technical endpoint/model details behind an expandable section.

Move embedder management into a clearly labeled “PDF search” advanced area. If PDF search has not been prepared, guide that setup separately; cloud chat still depends on source retrieval being ready. Do not silently change how already indexed PDFs are searched.

## Implementation sequence

### A. Connection state and reliable credentials

Introduce a provider-connection record containing a stable connection ID, provider ID, endpoint, credential reference, and connection state. A model references this connection rather than owning another copy of the key. Allow multiple custom endpoints/accounts by connection ID; do not globally key credentials only by brand.

Keep raw credentials in the main process after entry. Use a protected credential store, with Electron OS-backed encryption or a native credential-store adapter chosen and verified for macOS, Windows, and Linux. Persist only encrypted bytes/references in the appropriate protected location. Expose connection status, not raw keys, back to the renderer. Session-only behavior is explicit when persistent protection is unavailable.

Migration must preserve existing model identifiers and expose “Reconnect required” rather than filtering out remote models. A credential's presence means configured, not necessarily recently verified. Track transient network problems separately from authentication failure.

Relevant existing files:

- `src/main/engine/remote-model-secrets.ts`: current in-memory storage and redaction.
- `src/shared/app-state.ts`: model identity/status and proposed connection references.
- `src/main/index.ts`: load/save and connection IPC.
- `src/preload/index.ts` and shared bridge types: status/discovery/connect/reconnect contracts.

### B. One reusable connection flow

Extract the model picker, provider chooser, and cloud connection sheet from the large `App.tsx`. Mount the flow at the application level and pass an origin/return target so chat and Models share the same path.

Use an explicit progression: provider → credentials → discovering → model selection → verifying → ready, with recoverable errors attached to the relevant step. Preserve in-memory drafts across opening a help page and ordinary navigation; clear submitted/cancelled secret drafts deliberately.

Relevant code: `ModelsScreen` and its `remoteDrafts`, `loadRemoteProviderModels`, `installRemoteProviderModel`, `renderRemoteProviderCard`; the chat picker near `Choose chat model`; `upsertModel` navigation/selection behavior.

### C. Discovery and capability checks

Reuse the existing provider catalog and model-list service behind the new connection API. Fix role filtering: the current `listedModelMatchesRole` filters Gemini by name and accepts every listed model for other providers. Use supported provider metadata/adapters and a tested chat shortlist, with manual entry as an advanced fallback. Avoid claiming universal compatibility from `/models` alone.

Add typed connection errors, bounded discovery requests, and a minimal chat check compatible with each supported provider. The final connection must be durable before marking setup complete. Retry/idempotency must avoid duplicate model entries and preserve the previous working selection on failure.

Relevant files: `src/shared/model-providers.ts`, `src/main/engine/remote-chat-service.ts`, and the preload/main bridge.

### D. Finish and verify the student journey

Replace the cloud provider form wall, expose entry from chat, implement visible reconnect states, and streamline Models management. Keep custom endpoint and embedding controls reachable for advanced use.

Acceptance checks:

1. A student with an existing API key can connect and select a cloud model entirely from chat, without entering an endpoint or model ID for a supported provider.
2. Returning from the key-help page, cancelling, retrying, or reopening setup does not lose the current question or selected sources.
3. Restarting preserves protected provider access; missing credentials produce a visible reconnect state rather than a disappearing model.
4. A second model from the same connection reuses credentials; reconnecting updates all associated models.
5. Invalid key, unsupported model, quota failure, offline state, catalog failure, secure-store failure, and retired model all have specific recovery paths.
6. Cloud setup never changes the embedder or reindexes collections; local chat still works independently.
7. Keyboard navigation, Escape, focus return, readable contrast, and small-window layouts work throughout.
8. No key appears in app-state JSON, logs, errors, screenshots, or renderer responses after submission; validation never uses study material.
9. Automatic cloud requests obey the disclosed behavior, and no failed setup silently changes the service receiving the student's material.

Use focused state-machine and credential lifecycle tests, mocked provider responses, and a clean-profile UI walkthrough. Real provider checks belong in an opt-in integration test with a dedicated test key, never a screenshot-only claim of success.

Pilot with five students who already have a key. Target at least four completing setup unaided in under one minute, excluding provider signup/key creation and exceptional network delays. Treat this as a proposed usability target, not a measured result.

## Later, if justified

Consider an Ollama Cloud sign-in path for students already using Ollama, using its actual supported authentication flow. Treat this as a separate integration decision; a local Ollama endpoint may still route a cloud model remotely, so local/cloud labels must reflect execution location rather than endpoint host alone. School-managed access could remove the API-key step more completely, but requires a separate provisioning and billing design.

## Review materials

The accompanying interactive proposal illustrates the picker → provider → key → model → return-to-chat flow. It uses demo credentials and illustrative model choices; no network calls or actual credentials are accepted. Its appearance controls allow comparison of a top-bar versus composer picker and simulated connection outcomes. It is not an implementation of these services.

Prototype verification: walked the simulated Gemini connection through provider selection, demo key entry, discovery, model selection, and return to chat. The draft question and selected PDF remained present. A simulated quota failure displayed an actionable error while keeping the existing local model selected. Inspected desktop, 736-pixel, and 360-pixel layouts, including a dark error state; the 360-pixel view had no horizontal overflow. The final embedded script passes a syntax check. Actual provider authentication, OS credential storage, restart recovery, and production accessibility behavior remain implementation acceptance checks.
