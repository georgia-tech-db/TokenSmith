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
