"""
Offline pipeline to:
1. Generate dense section summaries using a large "Thinking" model.
2. Build a SQLite navigation index with section/paragraph one-line summaries.

Artifacts:
- Input:  data/extracted_sections.json
- Output: data/section_summaries.json
          data/nav_index.sqlite3
"""

import json
import re
import sqlite3
from pathlib import Path
from typing import List, Dict

from llama_cpp import Llama

# ---------- Paths / Config ----------

_SCRIPT_DIR = Path(__file__).parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent

EXTRACTED_SECTIONS_PATH = _PROJECT_ROOT / "data" / "extracted_sections.json"
SECTION_SUMMARIES_PATH = _PROJECT_ROOT / "data" / "section_summaries.json"
NAV_DB_PATH = _PROJECT_ROOT / "data" / "nav_index.sqlite3"

MODEL_PATH = str(_PROJECT_ROOT / "models" / "Qwen3-30B-A3B-Q6_K.gguf")

CTX_SIZE = 40960          # matches earlier summarizer
SUMMARY_BUDGET = 500      # tokens for dense section summary
ONE_LINE_MAX_TOKENS = 64  # for single-sentence summaries
ONE_LINE_MAX_CHARS = 200  # hard cap on character length


# ---------- Thinking Summarizer (integrated) ----------

class ThinkingSummarizer:
    """
    Large-model summarizer.

    - summarize_recursive(): dense section summary using rolling recursion.
    - process_section(): rolling paragraph summarization with buffer.
    - one_line(): single-sentence description using a separate prompt.
    """

    def __init__(self, model_path: str, n_ctx: int):
        print(f"[summ] Loading model: {model_path}")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=-1,
            tensor_split=[24, 24],  # adjust for your GPUs
            verbose=False,
            n_batch=512,
        )

    # ----- shared cleanup for thinking tags -----

    def clean_thinking_tokens(self, text: str) -> str:
        """Strip <think>/<reasoning> blocks while keeping the final answer."""
        if not text:
            return ""

        flags = re.DOTALL | re.IGNORECASE

        # If there is a closing </think>, keep everything AFTER the last one
        m = re.search(r'</think>\s*(.*)$', text, flags=flags)
        if m:
            text = m.group(1)

        # Same idea for </thinking> and </reasoning>
        for tag in ("thinking", "reasoning"):
            m = re.search(rf'</{tag}>\s*(.*)$', text, flags=flags)
            if m:
                text = m.group(1)

        # Remove remaining bare tags
        text = re.sub(r'</?think[^>]*>', '', text, flags=flags)
        text = re.sub(r'</?thinking[^>]*>', '', text, flags=flags)
        text = re.sub(r'</?reasoning[^>]*>', '', text, flags=flags)

        text = re.sub(r'\n\s*\n+', '\n\n', text)
        cleaned = text.strip()

        # If we somehow end up empty, treat as an error instead of hiding it
        if not cleaned:
            raise RuntimeError("LLM returned empty text after cleaning thinking tokens.")
        return cleaned

    # ----- dense rolling summary over long text -----

    def generate_update(self, current_summary: str, new_text: str) -> str:
        """
        Update the running summary with new information, staying within SUMMARY_BUDGET.
        """
        prompt = f"""<|im_start|>system
You are an expert technical summarizer. You are maintaining a dense, running summary of a technical document.
<|im_end|>
<|im_start|>user
Current Summary:
{current_summary or "(None)"}

New Content to Incorporate:
{new_text}

Task: Update the 'Current Summary' to include key information from 'New Content'.
Constraints:
1. Keep the total length under {SUMMARY_BUDGET} tokens.
2. Do not lose previous key details.
3. Output ONLY the updated summary. Do NOT include any thinking tags, reasoning blocks, or meta-commentary.
<|im_end|>
<|im_start|>assistant
"""
        output = self.llm.create_completion(
            prompt,
            max_tokens=SUMMARY_BUDGET + 100,
            temperature=0.3,
            stop=["<|im_end|>"],
        )
        raw = output["choices"][0]["text"]
        return self.clean_thinking_tokens(raw)

    def summarize_recursive(self, text: str, current_summary: str = "") -> str:
        """
        Recursively process text chunks.
        1. If text + summary fits context, update summary.
        2. If not, split text and recurse.
        """
        est_tokens = (len(text) + len(current_summary)) / 3.5

        if est_tokens < (CTX_SIZE - SUMMARY_BUDGET - 1000):
            return self.generate_update(current_summary, text)

        # Split roughly in half on a sentence boundary
        mid = len(text) // 2
        split_match = re.search(r'[.!?]\s', text[mid:])
        if split_match:
            split_idx = mid + split_match.end()
        else:
            split_idx = mid

        part1 = text[:split_idx]
        part2 = text[split_idx:]

        updated_summary = self.summarize_recursive(part1, current_summary)
        final_summary = self.summarize_recursive(part2, updated_summary)
        return final_summary

    def process_section(self, section_text: str) -> str:
        """
        Reads a section paragraph-by-paragraph to build a rolling summary.
        """
        paragraphs = section_text.split("\n\n")
        running_summary = ""
        buffer = ""

        for i, para in enumerate(paragraphs):
            para = para.strip()
            if not para:
                continue

            buffer += "\n" + para

            if len(buffer) > 8000:  # ~2200 tokens
                print(f"  [summ] Processing chunk {i + 1}/{len(paragraphs)}...")
                running_summary = self.summarize_recursive(buffer, running_summary)
                buffer = ""

        if buffer:
            running_summary = self.summarize_recursive(buffer, running_summary)

        return self.clean_thinking_tokens(running_summary)

    # ----- one-line summary -----

    def one_line(self, text: str) -> str:
        """
        Ask the model for a single-sentence one-liner.

        No fallback. If the call fails or yields nothing, an exception is raised.
        """
        if not text.strip():
            raise ValueError("Cannot create one-line summary from empty text.")

        prompt = f"""<|im_start|>system
You write concise, single-sentence descriptions of technical text.
Output exactly one sentence, under {ONE_LINE_MAX_CHARS} characters.
No bullet points, no lists, no markdown, no explanations.
<|im_end|>
<|im_start|>user
Text:
{text}
<|im_end|>
<|im_start|>assistant
"""
        output = self.llm.create_completion(
            prompt,
            max_tokens=ONE_LINE_MAX_TOKENS,
            temperature=0.3,
            stop=["<|im_end|>"],
        )
        raw = output["choices"][0]["text"]
        cleaned = self.clean_thinking_tokens(raw)
        cleaned = " ".join(cleaned.split())

        if len(cleaned) > ONE_LINE_MAX_CHARS:
            cleaned = cleaned[: ONE_LINE_MAX_CHARS - 3] + "..."

        if not cleaned:
            raise RuntimeError("LLM returned empty one-line summary.")
        return cleaned


# ---------- JSON Helpers ----------

def load_sections() -> List[Dict]:
    """Load raw sections with heading + content."""
    with open(EXTRACTED_SECTIONS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def split_paragraphs(text: str) -> List[str]:
    """Simple paragraph splitter using double newlines."""
    paras = [p.strip() for p in text.split("\n\n")]
    return [p for p in paras if p]


# ---------- SQLite Schema ----------

def init_db(conn: sqlite3.Connection) -> None:
    """Create tables, dropping any existing ones."""
    cur = conn.cursor()
    cur.executescript(
        """
        DROP TABLE IF EXISTS paragraphs;
        DROP TABLE IF EXISTS sections;

        CREATE TABLE sections (
            id               INTEGER PRIMARY KEY,
            heading          TEXT NOT NULL,
            section_summary  TEXT,
            one_line_summary TEXT,
            prev_section_id  INTEGER,
            next_section_id  INTEGER,
            num_paragraphs   INTEGER,
            content_length   INTEGER
        );

        CREATE TABLE paragraphs (
            id               INTEGER PRIMARY KEY,
            section_id       INTEGER NOT NULL,
            para_index       INTEGER NOT NULL,
            one_line_summary TEXT,
            raw_text         TEXT,
            FOREIGN KEY(section_id) REFERENCES sections(id)
        );

        CREATE INDEX idx_paragraphs_section
            ON paragraphs(section_id, para_index);
        """
    )
    conn.commit()

# ------------------ Reprocessing Single Section ------------------

def reprocess_single_section(section_id: int) -> None:
    """
    Re-run summarization + navigation indexing for a single section id.

    Steps:
      1) Recompute dense summary for the section.
      2) Recompute one-line summary for the section.
      3) Recompute one-line summaries for all paragraphs in that section.
      4) Update nav_index.sqlite3 (sections + paragraphs for this section only).
      5) Update section_summaries.json for this section only.

    Assumes:
      - extracted_sections.json is the source of truth.
      - nav_index.sqlite3 and section_summaries.json already exist and cover all sections.
    """
    # --- Load base sections (source of truth) ---
    if not EXTRACTED_SECTIONS_PATH.exists():
        raise FileNotFoundError(
            f"{EXTRACTED_SECTIONS_PATH} not found. Run extraction first."
        )
    sections = load_sections()
    total = len(sections)

    if section_id < 1 or section_id > total:
        raise ValueError(f"section_id {section_id} out of range (1..{total})")

    sec = sections[section_id - 1]
    heading = sec.get("heading", "Untitled")
    content = sec.get("content", "") or ""
    content_length = len(content)

    print(f"\n[reprocess] Section {section_id}/{total}: {heading} ({content_length} chars)")

    # --- Initialize summarizer (big model) ---
    summarizer = ThinkingSummarizer(MODEL_PATH, n_ctx=CTX_SIZE)

    if not content.strip():
        raise ValueError(f"Section {section_id} has no content; nothing to reprocess.")

    # --- 1) Dense section summary ---
    section_summary = summarizer.process_section(content)

    # --- 2) One-line section summary (from dense summary) ---
    one_line_section = summarizer.one_line(section_summary)

    # --- 3) Paragraphs + their one-liners ---
    paras = split_paragraphs(content)
    num_paras = len(paras)

    # prev / next IDs are purely sequential, as in the original pipeline
    prev_id = section_id - 1 if section_id > 1 else None
    next_id = section_id + 1 if section_id < total else None

    # --- 4) Update SQLite nav index (no schema reset) ---
    if not NAV_DB_PATH.exists():
        raise FileNotFoundError(
            f"{NAV_DB_PATH} not found. Run the full nav-index pipeline once first."
        )

    conn = sqlite3.connect(NAV_DB_PATH)
    cur = conn.cursor()

    # Make sure the expected tables exist
    cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='sections';"
    )
    if cur.fetchone() is None:
        conn.close()
        raise RuntimeError("SQLite DB missing 'sections' table.")

    cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='paragraphs';"
    )
    if cur.fetchone() is None:
        conn.close()
        raise RuntimeError("SQLite DB missing 'paragraphs' table.")

    # Delete existing rows for this section
    cur.execute("DELETE FROM paragraphs WHERE section_id = ?", (section_id,))
    cur.execute("DELETE FROM sections WHERE id = ?", (section_id,))

    # Insert updated section row
    cur.execute(
        """
        INSERT INTO sections (
            id, heading, section_summary, one_line_summary,
            prev_section_id, next_section_id, num_paragraphs, content_length
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            section_id,
            heading,
            section_summary,
            one_line_section,
            prev_id,
            next_id,
            num_paras,
            content_length,
        ),
    )

    # Insert updated paragraphs
    for p_idx, para_text in enumerate(paras):
        para_one_line = summarizer.one_line(para_text)
        cur.execute(
            """
            INSERT INTO paragraphs (
                section_id, para_index, one_line_summary, raw_text
            ) VALUES (?, ?, ?, ?)
            """,
            (section_id, p_idx, para_one_line, para_text),
        )

    conn.commit()
    conn.close()

    # --- 5) Update section_summaries.json entry for this section ---
    if not SECTION_SUMMARIES_PATH.exists():
        raise FileNotFoundError(
            f"{SECTION_SUMMARIES_PATH} not found. Run the full summary pipeline once first."
        )

    with open(SECTION_SUMMARIES_PATH, "r", encoding="utf-8") as f:
        summaries_data = json.load(f)

    # Basic sanity: expect at least 'total' entries and matching index
    if not isinstance(summaries_data, list):
        raise RuntimeError("section_summaries.json is not a list.")

    if len(summaries_data) < total:
        raise RuntimeError(
            f"section_summaries.json only has {len(summaries_data)} entries, "
            f"but extracted_sections has {total}."
        )

    summaries_data[section_id - 1] = {
        "heading": heading,
        "summary": section_summary,
        "content_length": content_length,
    }

    with open(SECTION_SUMMARIES_PATH, "w", encoding="utf-8") as f:
        json.dump(summaries_data, f, indent=2, ensure_ascii=False)

    print(f"[reprocess] Updated section {section_id} in nav_index.sqlite3 and section_summaries.json")


# ---------- Main Pipeline ----------

def main() -> None:
    if not EXTRACTED_SECTIONS_PATH.exists():
        raise FileNotFoundError(
            f"{EXTRACTED_SECTIONS_PATH} not found. Run extraction first."
        )

    NAV_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    sections = load_sections()
    total = len(sections)

    summarizer = ThinkingSummarizer(MODEL_PATH, n_ctx=CTX_SIZE)
    conn = sqlite3.connect(NAV_DB_PATH)
    init_db(conn)
    cur = conn.cursor()

    section_summaries_output: List[Dict] = []

    for idx, sec in enumerate(sections, start=1):
        heading = sec.get("heading", "Untitled")
        content = sec.get("content", "") or ""
        content_length = len(content)

        print(f"\n[{idx}/{total}] Section: {heading} ({content_length} chars)")

        if not content.strip():
            section_summary = ""
            one_line_section = ""
            num_paras = 0
        else:
            # 1) Dense section summary (rolling, recursive)
            section_summary = summarizer.process_section(content)

            # 2) One-line section summary from the dense summary
            one_line_section = summarizer.one_line(section_summary)

            # 3) Paragraphs + their one-liners
            paras = split_paragraphs(content)
            num_paras = len(paras)

        # prev / next IDs are purely sequential
        prev_id = idx - 1 if idx > 1 else None
        next_id = idx + 1 if idx < total else None

        # Insert section row
        cur.execute(
            """
            INSERT INTO sections (
                id, heading, section_summary, one_line_summary,
                prev_section_id, next_section_id, num_paragraphs, content_length
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                idx,
                heading,
                section_summary,
                one_line_section,
                prev_id,
                next_id,
                num_paras,
                content_length,
            ),
        )

        # Insert paragraphs
        if content.strip():
            paras = split_paragraphs(content)
            for p_idx, para_text in enumerate(paras):
                para_one_line = summarizer.one_line(para_text)
                cur.execute(
                    """
                    INSERT INTO paragraphs (
                        section_id, para_index, one_line_summary, raw_text
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (idx, p_idx, para_one_line, para_text),
                )

        conn.commit()

        # Collect for section_summaries.json
        section_summaries_output.append(
            {
                "heading": heading,
                "summary": section_summary,
                "content_length": content_length,
            }
        )

        # Incremental save of summaries file
        with open(SECTION_SUMMARIES_PATH, "w", encoding="utf-8") as f:
            json.dump(section_summaries_output, f, indent=2, ensure_ascii=False)

    conn.close()

    print(f"\n[done] Section summaries written to {SECTION_SUMMARIES_PATH}")
    print(f"[done] Navigation index written to {NAV_DB_PATH}")


if __name__ == "__main__":
    # reprocess_single_section(42)
    main()
