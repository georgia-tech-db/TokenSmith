# Chat Interaction Checks

Run `npx vite --config tests/ui/vite.config.mjs --host 127.0.0.1 --port 5186`, then open
`http://127.0.0.1:5186/tests/ui/chat-interactions.html`.

This renders the real app with a disposable, in-memory bridge and deterministic replies.
It does not contact a model or read/write the user's app data. Reload resets all test chats.

- Highlight part of a question or answer, including text across inline formatting. Add it to chat; confirm the quote is present and an existing draft is preserved.
- Select across two messages or inside a source/suggestion/editor. The selection action should not appear.
- Jump to each question using the rail; verify the preview, active tick, and return-to-latest button. Scroll up while a reply arrives; the view should not jump down.
- Open an older question's editor, modify it, then Cancel or Escape. The original question, later turns, and composer draft must remain intact.
- Save an older question. Cancel the confirmation once, then accept. Only the earlier history and replacement question should reach the test bridge; verify the message IDs in its reply.
- Edit the first and final questions. Neither should be duplicated. Test blank edits, multiline text, and switching chats while editing.
- Use `?long=1` for 80 questions.
- Open `chat-interactions-frame.html` to check 1280px, 866px and 390px layouts. Select the populated chat at 1280px, then change Preview width. The long-chat parameter works with the frame too.

The pure quote/edit transformations and edited-history pipeline contracts also run in
`npm run test:unit:ts` via `chat-interactions.test.mjs`.

For the clipped-selection regression, open `/tests/ui/selection-regression.html` with
the same preview server. “Check final selection” changes the browser range immediately
before clicking Add to chat, before the pending selection-change frame. “Check unfinished
drag” verifies the action stays hidden while the pointer is down. Both must show PASS
and insert the complete highlighted sentence while preserving the draft. The production
component failed both checks before the fix.

## Cloud generator setup

With the same preview server, open `/tests/ui/cloud-generators.html`. This uses the production App, picker, and connection sheet with a disposable fake bridge. Use demo strings only; no real key or network request is needed.

- Open the chat model picker, choose “Connect a cloud model,” select a provider, enter `demo-key`, choose a model, and use it. Verify the draft, previous messages, and selected book remain, with focus returning to the composer. Nomic must stay out of the chat picker.
- Reopen the provider to reuse its connection. Change its key to `bad-key` to check invalid-key recovery. A saved disconnected model must remain available through “Reconnect.”
- Add `?error=quota` or `?error=offline` for failure recovery; the existing model must remain selected. Add `?manual` for an empty catalog, `?session` for unavailable secure storage, or `?empty` for no generator.
- Use `slow-key`, then Back or Escape while discovery is pending; the delayed result must not reopen setup or select a model.
- Check keyboard focus, Escape and narrow-window layout. These fixtures do not verify OS storage or real provider compatibility; the cloud service unit tests and opt-in native checks cover those separately.
