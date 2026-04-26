# TokenSmith — Ed Discussion Auto-Answerer (Browser Extension)

A Chrome extension that automatically answers unanswered student questions on [Ed Discussion](https://edstem.org) using the local TokenSmith RAG server. It fetches every unanswered question thread from a selected course, sends it to your TokenSmith instance for AI-generated answers, and posts the responses back to Ed — all in one click.

---

## What It Does

1. **Detects your Ed session** — When you open the extension popup while on an `edstem.org` tab, it automatically reads your JWT authentication token from the page's `localStorage`. No manual token copy-pasting required.
2. **Lists your courses** — Populates a dropdown with every course where you hold the **admin** role.
3. **Finds unanswered questions** — Fetches all discussion threads for the selected course and filters down to threads of type `question` that have not yet been marked as answered.
4. **Generates answers via TokenSmith** — Sends the title and body of each unanswered question to the local TokenSmith `/api/chat` endpoint, which uses its RAG pipeline (retrieval over course materials, re-ranking, and LLM generation) to produce an answer.
5. **Posts answers back to Ed** — Submits each generated answer as an **answer-type comment** on the original Ed thread using Ed's API.

---

## How It Works

### Architecture

```
┌──────────────────────┐       ┌──────────────────────┐       ┌──────────────────┐
│   Ed Discussion Tab  │──────▶│   Chrome Extension   │──────▶│ TokenSmith Server│
│   (edstem.org)       │ JWT   │   (popup.js)         │ POST  │ (localhost:8000)  │
│                      │◀──────│                      │◀──────│                  │
│                      │ Post  │                      │Answer │                  │
└──────────────────────┘       └──────────────────────┘       └──────────────────┘
```

### Step-by-Step Flow

1. **Popup opens** → The extension checks if the active browser tab is on `edstem.org`.
2. **Token extraction** → If yes, it injects a script into the Ed tab (using `chrome.scripting.executeScript` in the `MAIN` world) to read `localStorage.getItem("authToken")` and `localStorage.getItem("authRegion")`. This gives it the user's JWT and their Ed region (e.g. `us`, `au`).
3. **API base URL** → The region is used to construct the correct API endpoint (e.g. `https://us.edstem.org/api`).
4. **User & courses** → A `GET /api/user` call retrieves the logged-in user's profile and enrolled courses. The dropdown is populated with courses where `role === "admin"`.
5. **Fetch threads** → On clicking "Answer Unanswered Questions", it paginates through `GET /api/courses/{id}/threads` (100 per page) and filters to `type === "question" && is_answered === false`.
6. **Query TokenSmith** → For each question, the title and body (with XML tags stripped) are sent as a `POST` to `{tokenSmithUrl}/api/chat` with payload `{ "query": "..." }`.
7. **Post answer** → The returned answer text is wrapped in Ed's XML document format (`<document version="2.0"><paragraph>...</paragraph></document>`) and posted via `POST /api/threads/{id}/comments` with type `"answer"`.

### Fallback Mode

If the user is **not** on an Ed tab, the extension falls back to **manual token mode**: a text field appears where you can paste an Ed API token directly. Clicking "Load Courses" then authenticates and populates the course list.

---

## ⚠️ Security Considerations — Who Can Use This

### Anyone with access to Ed can use it, not only admins

The extension currently filters the course dropdown to **admin-role** courses, but this is a **client-side filter only**. It provides no real access control:

- **Any Ed user** can install this extension. The JWT token it extracts authenticates as *whoever is logged in*, regardless of their role.
- **Students and TAs** can modify the extension source code (e.g. change the role filter from `"admin"` to `"student"` or remove the filter entirely) to list all their enrolled courses and trigger auto-answering on any of them.
- The extension runs entirely on the client side — there is no server-side authorization check on the TokenSmith end either. The `/api/chat` endpoint does not verify who is calling it or whether they should be allowed to post answers.

### The extension is trivially tamperable

Because Chrome extensions loaded in **Developer Mode** are just local files on disk:

- Anyone can open `popup.js`, remove the `role === "admin"` filter on **line 159**, and reload the extension to bypass the admin-only restriction.
- The Ed API token extracted from `localStorage` works for any API call the logged-in user is authorized to make on Ed — including posting answers, editing threads, etc.
- There is **no code signing, obfuscation, or integrity check** on the extension files.

### Implications

| Risk | Description |
|------|-------------|
| **Unauthorized posting** | A student could use this to auto-post AI-generated answers on threads they have access to. |
| **Token exposure** | The JWT is extracted from the browser and used for API calls. If the extension is modified maliciously, the token could be exfiltrated. |
| **No audit trail** | Answers are posted under the logged-in user's Ed account. There is no indication that the answer was AI-generated. |
| **TokenSmith is open** | The local TokenSmith server (`localhost:8000`) has no authentication, so any local process or extension can query it. |

> **Bottom line:** Treat this extension as a convenience tool for trusted users, not a secured system. If you need access control, implement server-side authorization on the TokenSmith API and/or use a dedicated service account for Ed API interactions.

---

## How to Load the Extension into Chrome

### Prerequisites

- **Google Chrome** (or any Chromium-based browser: Edge, Brave, Arc, etc.)
- **TokenSmith server** running locally (default: `http://localhost:8000`)
- An **Ed Discussion account** with at least one course

### Installation Steps

1. **Clone or download** the `browser-extension/` folder from this repository:
   ```bash
   git clone https://github.com/georgia-tech-db/TokenSmith.git
   cd TokenSmith/browser-extension
   ```

2. **Open Chrome's extension management page:**
   - Navigate to `chrome://extensions/` in the address bar
   - Or go to **⋮ Menu → Extensions → Manage Extensions**

3. **Enable Developer Mode:**
   - Toggle the **"Developer mode"** switch in the top-right corner of the extensions page

4. **Load the extension:**
   - Click **"Load unpacked"**
   - Select the `browser-extension/` folder (the one containing `manifest.json`)
   - The extension icon should appear in your Chrome toolbar

5. **Pin the extension** (optional but recommended):
   - Click the puzzle-piece icon (🧩) in the toolbar
   - Pin **"TokenSmith — Ed Discussion Auto-Answerer"** for easy access

---

## Using the Extension with TokenSmith Server

### 1. Start the TokenSmith Server

Make sure your TokenSmith server is running before using the extension:

```bash
cd TokenSmith
python -m src.main
# Server starts on http://localhost:8000 by default
```

### 2. Navigate to Ed Discussion

Open any page on [edstem.org](https://edstem.org) and make sure you are logged in.

### 3. Open the Extension Popup

Click the TokenSmith extension icon in your toolbar. You should see:

- A **green dot** with "Signed in as [Your Name]" — the extension auto-detected your Ed session
- A **course dropdown** populated with your admin courses
- The **TokenSmith Server URL** field pre-filled with `http://localhost:8000`

### 4. Configure (if needed)

- **TokenSmith URL** — If your server runs on a different host or port, update the URL field (e.g. `http://192.168.1.50:8000`)
- **Manual token mode** — If you're not on an Ed tab, click "Use manual API token instead", paste your Ed API token, and click "Load Courses"

### 5. Run Auto-Answering

1. Select a course from the dropdown
2. Click **🚀 Answer Unanswered Questions**
3. Watch the log panel for real-time progress:
   - `⏳ Fetching unanswered questions…`
   - `[1/5] #42: How does virtual memory work?`
   - `⏳ Querying TokenSmith…`
   - `✓ Answer posted!`
4. When complete, all unanswered questions will have AI-generated answers posted

---

## File Structure

```
browser-extension/
├── manifest.json    # Chrome Extension Manifest V3 — permissions, icons, popup
├── popup.html       # Extension popup UI — styled dark-theme interface
├── popup.js         # All extension logic — auth, API calls, orchestration
├── icon16.png       # Toolbar icon (16×16)
├── icon48.png       # Extension page icon (48×48)
├── icon128.png      # Chrome Web Store icon (128×128)
└── README.md        # This file
```

---

## Permissions

The extension requests the following Chrome permissions (defined in `manifest.json`):

| Permission | Why |
|---|---|
| `storage` | Persist the Ed API token and TokenSmith URL between sessions |
| `activeTab` | Read the URL of the active tab to detect if it's an Ed page |
| `scripting` | Inject a script into the Ed tab to extract the JWT from `localStorage` |
| `host_permissions: edstem.org/*` | Make API requests to Ed Discussion (us, au, and base domains) |
| `host_permissions: localhost:8000/*` | Send queries to the local TokenSmith server |
