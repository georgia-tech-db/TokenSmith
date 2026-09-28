# Chat Interaction Checks

Run `npx vite --config tests/ui/vite.config.mjs --host 127.0.0.1 --port 5186`, then open
`http://127.0.0.1:5186/tests/ui/chat-interactions.html`.

This renders the real app with a disposable, in-memory bridge and deterministic replies.
It does not contact a model or read/write the user's app data. Reload resets all test chats.

- Highlight part of a question or answer, including text across inline formatting. Click Ask about this; confirm a separate passage preview appears, the existing draft is unchanged, and the question field receives focus. Selection alone must not enable Send.
- Remove the passage; confirm the draft remains. Select a different passage; confirm it replaces the preview. Switch chats and back; drafts and selections must stay with their own chat.
- Send a question with a passage. The test reply must report the exact passage, original question, and source message ID separately from the question. Edit that question; keep or remove the passage, cancel once, then resend. Cancel must leave the saved selection unchanged.
- Select across two messages or inside a source/suggestion/editor. The selection action should not appear.
- Jump to each question using the rail; verify the preview, active tick, and return-to-latest button. Scroll up while a reply arrives; the view should not jump down.
- Open an older question's editor, modify it, then Cancel or Escape. The original question, later turns, and composer draft must remain intact.
- Save an older question. Cancel the confirmation once, then accept. Only the earlier history and replacement question should reach the test bridge; verify the message IDs in its reply.
- Edit the first and final questions. Neither should be duplicated. Test blank edits, multiline text, and switching chats while editing.
- Use `?long=1` for 80 questions.
- Open `chat-interactions-frame.html` to check 1280px, 866px and 390px layouts. Select the populated chat at 1280px, then change Preview width. The long-chat parameter works with the frame too.

The selection/edit transformations and edited-history pipeline contracts also run in
`npm run test:unit:ts` via `chat-interactions.test.mjs`.

For the clipped-selection regression, open `/tests/ui/selection-regression.html` with
the same preview server. “Check final selection” changes the browser range immediately
before clicking Ask about this, before the pending selection-change frame. “Check unfinished
drag” verifies the action stays hidden while the pointer is down. Both must show PASS
and attach the complete highlighted sentence while preserving the draft. The production
component failed both checks before the fix.

## Cloud generator setup

With the same preview server, open `/tests/ui/cloud-generators.html`. This uses the production App, picker, and connection sheet with a disposable fake bridge. Use demo strings only; no real key or network request is needed.

- Open the chat model picker, choose “Connect a cloud model,” select a provider, enter `demo-key`, choose a model, and use it. Verify the draft, previous messages, and selected book remain, with focus returning to the composer. Nomic must stay out of the chat picker.
- Reopen the provider to reuse its connection. Change its key to `bad-key` to check invalid-key recovery. A saved disconnected model must remain available through “Reconnect.”
- Add `?error=quota` or `?error=offline` for failure recovery; the existing model must remain selected. Add `?manual` for an empty catalog, `?session` for unavailable secure storage, or `?empty` for no generator.
- Use `slow-key`, then Back or Escape while discovery is pending; the delayed result must not reopen setup or select a model.
- Check keyboard focus, Escape and narrow-window layout. These fixtures do not verify OS storage or real provider compatibility; the cloud service unit tests and opt-in native checks cover those separately.

## Navigation while an answer is pending

Open `/tests/ui/chat-navigation.html` with the same preview server. This runs the production App with a fake engine; the test toolbar completes or rejects a request manually. No model or user files are used. The fixture keeps its own state in session storage, starting with the legacy `medium` font preference.

- Send a question, open Settings, change the font size, and return to Chat. The toolbar must still show exactly one call and one pending request, with **Stop** visible in the bottom status strip. Before the fix, the Stop button disappeared after navigation while the engine still had a pending request.
- Open Settings again and use **Finish pending answer**. Settings must stay visible; returning to Chat must show exactly one completed answer and no pending indicator.
- Repeat, stop the response after returning to Chat, then finish the pending test request. The late answer must be ignored. Use **Fail pending answer** while in Settings to verify the error is shown on return.
- Type an unsent draft and switch tabs; the draft must survive. Chat-only selection/hover overlays must not appear over Settings.
- Font size must list Extra small, Small, Normal, Large, Extra large. Chat text sizes are 12, 14, 16, 18, 20 px. The legacy Medium value becomes Normal. Change the size and reload; the selection must persist. There is no sample-text preview block.

## Generation status and learned estimates

Open `/tests/ui/generation-progress.html`. This uses the real App with manually controlled answer and follow-up phases, and fixture-only session data. The toolbar shows request, cancellation and timing-sample counts. Timing history is stored on the test origin, separate from desktop data.

- Send a question, open Settings or Library, and confirm one bottom status strip remains visible. Stop from Settings must cancel the fake request and add no timing sample.
- Use **Deliver answer** while in Settings. The strip switches to Preparing follow-ups without navigating away. Click its label to return to the conversation: the answer must already be readable while the request is still pending.
- **Finish suggestions** must update that same answer once, show its follow-ups, and briefly show Answer ready. Timing samples increase once for the answer and once for suggestions.
- Deliver an answer, then submit another question before completing its follow-ups. The old follow-up request must be canceled, the completed answer kept, and exactly one new request left pending. Stopping or failing the new request must not train the estimator.
- Reload and send another question using the same configuration; the estimate tooltip should say it uses recent runs. Choose another model or change thinking/output settings to verify a separate profile.
- Use `?starter`, then **Reset fixture**, to test automatic starter questions. They share the strip, cancel when leaving Chat, and train only after successful completion. Sending a question takes priority.
- Let a request exceed its displayed estimate: the bar must become indeterminate, show elapsed time, and never claim 100% or zero seconds remaining. Use `?fast-estimate` for a short seeded estimate.
- Check 866px and 390px widths, Extra large text, and reduced motion. The strip must stay below the workspace without covering the composer. A failed inference must clear it; stale completion must not replace a newer request's status.

These checks verify behavior, not prediction accuracy. Predictions improve through actual completed runs; no additional model calls are made to obtain estimates.

## Read the first source while waiting

Use `/tests/ui/generation-progress.html?preview=markdown` (or `?preview=pdf`) with the same preview server. These variants supply tiny fixture documents through the fake bridge; no user files are opened.

- Send a question. The first source opens automatically at its passage/page inside Chat, with navigation and the bottom progress strip still accessible.
- **Deliver answer** closes the automatic reader immediately while the strip continues Preparing follow-ups. **Finish suggestions** updates the same answer.
- Close the reader yourself during generation; it must remain closed. Stop or fail the request; the automatic reader must close. Visit Settings; it must not remain above Settings or reopen when the answer arrives.
- Add `&slow-source`, send, deliver the answer, then click **Finish source load**. The late load must not open a reader. Repeat with Stop and a conversation switch.
- Add `&answer-delay`, send, and dismiss the automatic reader. Click **Deliver answer**, then immediately open Second source manually. After the two-second answer delay, that manually opened source must remain visible. Close it and verify the answer is already present.
- Turn Show Sources off: no automatic reader should open. With no retrieved source or an unavailable document, answer generation must continue normally.

## Contents and mixed source navigation

Use `/tests/ui/generation-progress.html?preview=markdown&mixed`. This supplies two Markdown files and two PDFs, both as retrieved sources and as a Library collection.

- Expand Contents, click the second Details entry, and verify that it jumps to the second matching section. Summary uses Setext syntax and must appear; `# Not a heading` inside the code fence must not. Back to chunk returns to the highlighted passage. Collapse Contents, reopen another Markdown source, and confirm the preference is remembered.
- Next source visits PDF, Markdown, then PDF, showing Source 1–4 of 4. Previous visits the reverse order. Boundary buttons are disabled. PDF page navigation remains independent.
- Navigate to a different source while generation is pending, then Deliver answer. The chosen reader must stay open.
- In Library, open Course book, select slides.pdf, and open Page 1. It starts at Source 2 of 4; Next opens exercises.md and then appendix.pdf, regardless of which document's chunks are selected behind the reader.
- Add `&fail-source`: Next from the first Markdown source must keep that document visible and show the PDF load error. Add `&slow-source`: finish the first load, click Next, close the reader, then Finish source load. The late destination must not reopen the reader.
- Check 866px and 390px widths. The header controls remain visible; Contents becomes a panel above the text in narrow windows.

Parsed heading targets and mixed-source ordering also run in the TypeScript unit suite.
