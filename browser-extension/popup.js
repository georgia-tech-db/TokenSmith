/**
 * TokenSmith — Ed Discussion Auto-Answerer
 *
 * Single popup script that:
 *   1. Auto-detects the Ed session from the active edstem.org tab (no token needed).
 *   2. Falls back to manual API token entry if not on an Ed tab.
 *   3. Lists courses where the user has the "admin" role.
 *   4. Fetches all unanswered *questions* (ignoring announcements & posts).
 *   5. Sends each question to the local TokenSmith server (/api/chat).
 *   6. Posts the generated answer back to Ed as an "answer" comment.
 */

// ─── Constants ──────────────────────────────────────────────────────
let ED_API = "https://us.edstem.org/api";
const ADMIN_ROLE = "admin";
const THREADS_PER_PAGE = 100; // max Ed supports

// ─── DOM refs ───────────────────────────────────────────────────────
const $edToken         = document.getElementById("edToken");
const $tsUrl           = document.getElementById("tsUrl");
const $courseSelect    = document.getElementById("courseSelect");
const $btnLoad         = document.getElementById("btnLoad");
const $btnGo           = document.getElementById("btnGo");
const $btnFallback     = document.getElementById("btnFallback");
const $fallbackSection = document.getElementById("fallbackSection");
const $sessionBadge    = document.getElementById("sessionBadge");
const $sessionName     = document.getElementById("sessionName");
const $sessionSub      = document.getElementById("sessionSub");
const $log             = document.getElementById("log");

// ─── State ──────────────────────────────────────────────────────────
let courseMap = {};        // courseId → { name, role }
let authMode = "none";    // "session" | "token" | "none"
let edTabId  = null;      // tab ID for session mode

// ─── Helpers ────────────────────────────────────────────────────────

/** Persistent storage for settings. */
function saveSettings() {
  chrome.storage.local.set({
    edToken: $edToken.value.trim(),
    tsUrl:   $tsUrl.value.trim(),
  });
}

async function loadSettings() {
  return new Promise((resolve) => {
    chrome.storage.local.get(["edToken", "tsUrl"], (data) => {
      if (data.edToken) $edToken.value = data.edToken;
      if (data.tsUrl)   $tsUrl.value   = data.tsUrl;
      resolve();
    });
  });
}

/** Append a styled line to the log panel. */
function log(msg, cls = "") {
  $log.classList.add("visible");
  const span = document.createElement("span");
  span.className = cls;
  span.textContent = msg + "\n";
  $log.appendChild(span);
  $log.scrollTop = $log.scrollHeight;
}

// ─── Session badge UI helpers ───────────────────────────────────────

function setSessionBadge(state, name, sub) {
  const dot = $sessionBadge.querySelector(".dot");
  dot.className = `dot ${state}`;   // "green" | "yellow" | "red"
  $sessionName.textContent = name;
  $sessionSub.textContent  = sub;
}

function showTokenMode() {
  setSessionBadge("red", "Not on an Ed tab", "Enter a token manually below");
  $btnFallback.style.display = "block";
  $fallbackSection.classList.add("visible");
  $btnLoad.style.display = "block";
  $courseSelect.innerHTML = '<option value="">— enter token & load courses —</option>';
  authMode = "token";
}

function showSessionMode(userName, userEmail) {
  setSessionBadge("green", `Signed in as ${userName}`, userEmail);
  $btnFallback.style.display = "none";
  $fallbackSection.classList.remove("visible");
  $btnLoad.style.display = "none";
  authMode = "session";
}

// ─── Ed API transport layer ─────────────────────────────────────────

/** The active Bearer token — either grabbed from the Ed tab or entered manually. */
let edBearerToken = null;

/**
 * Session mode: inject a tiny script into the Ed tab to read the JWT
 * from localStorage. Ed stores it under the key "authToken".
 * We only need to do this once, then reuse the token for all requests.
 */
async function extractTokenFromTab() {
  const results = await chrome.scripting.executeScript({
    target: { tabId: edTabId },
    world: "MAIN",
    func: () => {
      return {
        token: localStorage.getItem("authToken") || null,
        region: localStorage.getItem("authRegion") || null,
      };
    },
  });

  const data = results?.[0]?.result;
  if (!data?.token) {
    throw new Error("No authToken found in Ed localStorage — are you logged in?");
  }
  return data;
}

/**
 * Unified Ed API fetch — uses X-Token auth header.
 * The token comes from either the Ed tab's localStorage or manual input.
 */
async function edFetch(path, options = {}) {
  const token = edBearerToken || $edToken.value.trim();
  if (!token) {
    throw new Error("No Ed API token available.");
  }

  const opts = {
    method: options.method || "GET",
    headers: {
      "X-Token": token,
      "Content-Type": "application/json",
    },
  };
  if (options.body) opts.body = options.body;

  const res = await fetch(`${ED_API}${path}`, opts);
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`Ed API ${res.status}: ${text.slice(0, 300)}`);
  }
  return res.json();
}

// ─── Core Logic ─────────────────────────────────────────────────────

/**
 * Populate the course dropdown from user data.
 * Filters to admin-role courses only.
 */
function populateCourses(data) {
  const user = data.user;

  const eligible = (data.courses || []).filter((entry) => {
    const role = entry.role?.role;
    return role === ADMIN_ROLE;
  });

  if (eligible.length === 0) {
    log("⚠ No courses found where you are an admin.", "log-warn");
    $courseSelect.innerHTML = '<option value="">— no admin courses —</option>';
    return;
  }

  $courseSelect.innerHTML = "";
  courseMap = {};

  eligible.forEach((entry) => {
    const c = entry.course;
    const role = entry.role.role;
    courseMap[c.id] = { name: `${c.code} – ${c.name}`, role };

    const opt = document.createElement("option");
    opt.value = c.id;
    opt.textContent = `${c.code} – ${c.name}  (${role})`;
    $courseSelect.appendChild(opt);
  });

  $courseSelect.disabled = false;
  $btnGo.disabled = false;
  log(`✓ Found ${eligible.length} admin course(s).`, "log-ok");
}

/**
 * Manual "Load Courses" for token mode.
 */
async function loadCourses() {
  if (!$edToken.value.trim()) {
    log("⚠ Please enter your Ed API token first.", "log-warn");
    return;
  }
  saveSettings();
  authMode = "token";
  log("⏳ Fetching user profile & courses…", "log-info");

  try {
    const data = await edFetch("/user");
    const user = data.user;
    setSessionBadge("green", `Signed in as ${user.name}`, `${user.email} (via token)`);
    log(`✓ Logged in as ${user.name} (${user.email})`, "log-ok");
    populateCourses(data);
  } catch (err) {
    log(`✗ ${err.message}`, "log-err");
  }
}

/**
 * List ALL threads for the selected course, paginating as needed.
 * Filter to: type === "question" && is_answered === false
 */
async function fetchUnansweredQuestions(courseId) {
  let allThreads = [];
  let offset = 0;

  // eslint-disable-next-line no-constant-condition
  while (true) {
    const data = await edFetch(
      `/courses/${courseId}/threads?limit=${THREADS_PER_PAGE}&offset=${offset}&sort=new`
    );
    const threads = data.threads || [];
    allThreads = allThreads.concat(threads);

    if (threads.length < THREADS_PER_PAGE) break; // last page
    offset += THREADS_PER_PAGE;
  }

  // Keep only unanswered questions (ignore announcements, posts, answered)
  return allThreads.filter(
    (t) => t.type === "question" && !t.is_answered
  );
}

/**
 * Send the question text to the local TokenSmith /api/chat endpoint.
 * Returns the generated answer string.
 */
async function askTokenSmith(questionText) {
  const tsBase = $tsUrl.value.trim().replace(/\/+$/, "");
  const res = await fetch(`${tsBase}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query: questionText }),
  });

  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`TokenSmith ${res.status}: ${text.slice(0, 200)}`);
  }

  const data = await res.json();
  return data.answer;
}

/**
 * Post an answer to an Ed thread.
 * POST /api/threads/<thread_id>/comments
 * Ed expects content in its XML "document" format.
 */
function wrapInEdXml(plainText) {
  const escaped = plainText
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");

  const paragraphs = escaped
    .split(/\n/)
    .map((line) => `<paragraph>${line}</paragraph>`)
    .join("");

  return `<document version="2.0">${paragraphs}</document>`;
}

async function postAnswer(threadId, answerText) {
  const content = wrapInEdXml(answerText);
  await edFetch(`/threads/${threadId}/comments`, {
    method: "POST",
    body: JSON.stringify({
      comment: {
        type: "answer",
        content,
      },
    }),
  });
}

/**
 * Main "Go" flow — orchestrate everything.
 */
async function runAutoAnswer() {
  const courseId = $courseSelect.value;
  if (!courseId) {
    log("⚠ Select a course first.", "log-warn");
    return;
  }
  saveSettings();

  const courseName = courseMap[courseId]?.name || courseId;
  log(`\n━━━ Starting for ${courseName} ━━━`, "log-info");

  // Disable buttons during run
  $btnGo.disabled = true;
  $btnLoad.disabled = true;
  $btnGo.innerHTML = '<span class="spinner"></span>&nbsp; Processing…';

  try {
    // Step A — get unanswered questions
    log("⏳ Fetching unanswered questions…", "log-info");
    const questions = await fetchUnansweredQuestions(courseId);

    if (questions.length === 0) {
      log("✓ No unanswered questions found — all caught up!", "log-ok");
      return;
    }

    log(`Found ${questions.length} unanswered question(s).`, "log-info");
    // Step B — process each question
    for (let i = 0; i < questions.length; i++) {
      const q = questions[i];
      const label = `[${i + 1}/${questions.length}]`;

      log(`${label} #${q.number}: ${q.title}`, "log-sub");

      // Extract plain text from the question body
      const questionText = `${q.title}\n\n${stripXmlTags(q.content || q.document || "")}`;

      try {
        log(`  ⏳ Querying TokenSmith…`, "log-info");
        const answer = await askTokenSmith(questionText);

        if (!answer || answer.trim().length === 0) {
          log(`  ⚠ TokenSmith returned an empty answer — skipping.`, "log-warn");
          continue;
        }
        log(`  ⏳ Posting answer to Ed…`, "log-info");
        await postAnswer(q.id, answer);
        log(`  ✓ Answer posted!`, "log-ok");
      } catch (err) {
        log(`  ✗ Error: ${err.message}`, "log-err");
      }
    }

    log(`\n✓ Done — processed ${questions.length} question(s).`, "log-ok");
  } catch (err) {
    log(`✗ ${err.message}`, "log-err");
  } finally {
    $btnGo.disabled = false;
    $btnLoad.disabled = false;
    $btnGo.innerHTML = "🚀&nbsp; Answer Unanswered Questions";
  }
}

/** Rough XML/HTML tag stripper. */
function stripXmlTags(str) {
  return str
    .replace(/<[^>]+>/g, " ")
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/\s+/g, " ")
    .trim();
}

// ─── Initialization ─────────────────────────────────────────────────

/**
 * On popup open:
 *   1. Check if the active tab is an edstem.org page.
 *   2. If yes → inject a script to call /api/user using the tab's session cookies.
 *   3. Auto-populate courses from the response.
 *   4. If not on Ed → fall back to manual token mode.
 */
async function init() {
  await loadSettings();

  // Check the active tab
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

  if (tab?.url && /edstem\.org/.test(tab.url)) {
    edTabId = tab.id;
    setSessionBadge("yellow", "Connecting to Ed session…", tab.url);

    try {
      // Grab the JWT + region from Ed's localStorage
      const { token, region } = await extractTokenFromTab();
      edBearerToken = token;
      authMode = "session";

      // Set API base using region (e.g. "us" → https://us.edstem.org/api)
      if (region) {
        ED_API = `https://${region}.edstem.org/api`;
      }
      log(`✓ Token extracted (region: ${region || "default"})`, "log-ok");

      // Now use the token to fetch user info & courses
      const data = await edFetch("/user");
      const user = data.user;

      showSessionMode(user.name, user.email);
      log(`✓ Session detected — ${user.name} (${user.email})`, "log-ok");
      populateCourses(data);
    } catch (err) {
      log(`⚠ Could not detect Ed session: ${err.message}`, "log-warn");
      log("Falling back to manual token mode.", "log-info");
      edTabId = null;
      edBearerToken = null;
      showTokenMode();
    }
  } else {
    // Not on an Ed tab
    showTokenMode();
    log("Open an edstem.org tab for automatic session detection.", "log-info");
  }
}

// ─── Event Listeners ────────────────────────────────────────────────
$btnLoad.addEventListener("click", loadCourses);
$btnGo.addEventListener("click", runAutoAnswer);
$btnFallback.addEventListener("click", () => {
  $fallbackSection.classList.toggle("visible");
});

// Start
init();
