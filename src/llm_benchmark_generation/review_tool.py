"""
src/llm_benchmark_generation/review_tool.py
============================================
Streamlit app for manual review and annotation of generated QAC pairs.

Run from the TokenSmith project root:
    streamlit run src/llm_benchmark_generation/review_tool.py

Requirements:
    pip install streamlit
"""

import json
import pathlib
import re
import time
from copy import deepcopy

import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Paths — all relative to project root
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT   = pathlib.Path(__file__).parent.parent.parent   # TokenSmith/
QAC_DIR        = PROJECT_ROOT / "synthetic_qac_data" / "qacs"
VERIFIED_DIR   = PROJECT_ROOT / "synthetic_qac_data" / "manually_verified"
MD_PATH        = PROJECT_ROOT / "data" / "textbook--extracted_markdown.md"

# ─────────────────────────────────────────────────────────────────────────────
# Annotation statuses
# ─────────────────────────────────────────────────────────────────────────────

STATUS_UNANNOTATED = "unannotated"
STATUS_APPROVED    = "approved"
STATUS_EDITED      = "edited_approved"
STATUS_REJECTED    = "rejected"
STATUS_SKIPPED     = "skipped"

ALL_ANNOTATION_STATUSES = [
    STATUS_UNANNOTATED,
    STATUS_APPROVED,
    STATUS_EDITED,
    STATUS_REJECTED,
    STATUS_SKIPPED,
]

DIFFICULTY_COLOURS = {
    "easy":   "#2d6a4f",
    "medium": "#b5770c",
    "hard":   "#a4262c",
}

ANNOTATION_COLOURS = {
    STATUS_UNANNOTATED: "#555555",
    STATUS_APPROVED:    "#2d6a4f",
    STATUS_EDITED:      "#1a5276",
    STATUS_REJECTED:    "#a4262c",
    STATUS_SKIPPED:     "#7d6608",
}

# ─────────────────────────────────────────────────────────────────────────────
# Markdown loading & substring verification
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_markdown() -> tuple[str, dict[int, int]]:
    """Load the full markdown once per session and build page offset index."""
    if not MD_PATH.exists():
        return "", {}
    with open(MD_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    text = re.sub(r"<!--\s*image\s*-->", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    offsets: dict[int, int] = {}
    for m in re.finditer(r"--- Page (\d+) ---", text):
        offsets[int(m.group(1))] = m.start()
    return text, offsets


def extract_pages(full_md: str, offsets: dict[int, int], start: int, end: int) -> str:
    if start not in offsets:
        return ""
    begin    = offsets[start]
    max_page = max(offsets.keys()) if offsets else 0
    next_page = end + 1
    while next_page <= max_page and next_page not in offsets:
        next_page += 1
    end_char = offsets.get(next_page, len(full_md))
    return full_md[begin:end_char]


def normalise_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def clean_md_for_verification(text: str) -> str:
    text    = re.sub(r"^\s*[\.,!?;:\-_]+\s*$", " ", text, flags=re.MULTILINE)
    cleaned = re.sub(r"---\s*Page\s+\d+\s*---", " ", text)
    cleaned = re.sub(r"\bPage\s+\d+\b", " ", cleaned)
    return normalise_ws(cleaned)


def check_chunk(chunk: str, pages_text: str) -> bool:
    norm_source   = normalise_ws(pages_text)
    clean_source  = clean_md_for_verification(pages_text)
    norm_chunk    = normalise_ws(chunk)
    norm_stripped = norm_chunk.rstrip(".,;:!?")
    return (
        norm_chunk    in norm_source
        or norm_chunk in clean_source
        or norm_stripped in norm_source
        or norm_stripped in clean_source
    )


# ─────────────────────────────────────────────────────────────────────────────
# File discovery
# ─────────────────────────────────────────────────────────────────────────────

def list_qac_files() -> list[pathlib.Path]:
    """Return all JSONL files in the qacs/ directory, sorted."""
    if not QAC_DIR.exists():
        return []
    return sorted(QAC_DIR.glob("*.jsonl"))


def get_verified_path(source_path: pathlib.Path) -> pathlib.Path:
    """Derive the output path for a given source JSONL file."""
    VERIFIED_DIR.mkdir(parents=True, exist_ok=True)
    return VERIFIED_DIR / f"Verified--{source_path.name}"


# ─────────────────────────────────────────────────────────────────────────────
# Data loading & saving
# ─────────────────────────────────────────────────────────────────────────────

def load_jsonl(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def load_verified_index(verified_path: pathlib.Path) -> dict[str, dict]:
    """Load existing annotations keyed by record_id."""
    annotations: dict[str, dict] = {}
    for rec in load_jsonl(verified_path):
        rid = rec.get("record_id")
        if rid:
            annotations[rid] = rec
    return annotations


def save_verified(records: list[dict], verified_path: pathlib.Path) -> None:
    verified_path.parent.mkdir(parents=True, exist_ok=True)
    with open(verified_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def make_record_id(rec: dict, idx: int) -> str:
    return (
        f"c{rec.get('chapter', 0):02d}_"
        f"w{rec.get('window_pages', [0,0])[0]}_"
        f"{rec.get('window_pages', [0,0])[1]}_"
        f"q{idx:03d}"
    )


def merge_records(
    generated: list[dict],
    verified:  dict[str, dict],
) -> list[dict]:
    """
    Merge generated records with existing annotations.
    New records start as unannotated. Existing annotations are fully restored.
    """
    merged = []
    for i, rec in enumerate(generated):
        rid  = make_record_id(rec, i)
        base = deepcopy(rec)
        base["record_id"] = rid

        if rid in verified:
            ann = verified[rid]
            base["annotation_status"]    = ann.get("annotation_status", STATUS_UNANNOTATED)
            base["annotator_note"]       = ann.get("annotator_note", "")
            base["annotation_timestamp"] = ann.get("annotation_timestamp", "")
            base["edited_fields"]        = ann.get("edited_fields", {})
            for field in ("question", "mock_answer", "rubric",
                          "gold_chunks", "chunk_relationships", "difficulty"):
                if field in ann.get("edited_fields", {}):
                    base[field] = ann["edited_fields"][field]
        else:
            base["annotation_status"]    = STATUS_UNANNOTATED
            base["annotator_note"]       = ""
            base["annotation_timestamp"] = ""
            base["edited_fields"]        = {}

        merged.append(base)
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────

def init_state() -> None:
    """Initialise session state keys that persist across reruns."""
    if "selected_file" not in st.session_state:
        st.session_state.selected_file = None   # pathlib.Path
    if "records" not in st.session_state:
        st.session_state.records       = []
    if "selected" not in st.session_state:
        st.session_state.selected      = None   # int index
    if "full_md" not in st.session_state:
        full_md, offsets = load_markdown()
        st.session_state.full_md = full_md
        st.session_state.offsets = offsets

    # ── Per-record dynamic list state ─────────────────────────────────────────
    # Rubric items and chunk items for the currently open record are kept in
    # session state so that "add" / "delete" buttons work across reruns.
    if "edit_rubric" not in st.session_state:
        st.session_state.edit_rubric = []    # list[str]
    if "edit_chunks" not in st.session_state:
        st.session_state.edit_chunks = []    # list[str]
    if "edit_record_idx" not in st.session_state:
        st.session_state.edit_record_idx = None  # tracks which record owns these lists


def load_file(path: pathlib.Path) -> None:
    """Load a QAC source file and its corresponding verified file into session state."""
    verified_path   = get_verified_path(path)
    generated       = load_jsonl(path)
    verified_index  = load_verified_index(verified_path)
    st.session_state.selected_file = path
    st.session_state.records       = merge_records(generated, verified_index)
    st.session_state.selected      = None
    st.session_state.edit_record_idx = None


def sync_edit_state(idx: int) -> None:
    """
    Sync the dynamic edit lists to the record at idx.
    Called whenever the selected record changes so the lists are fresh.
    """
    if st.session_state.edit_record_idx != idx:
        rec = st.session_state.records[idx]
        st.session_state.edit_rubric     = list(rec.get("rubric", []))
        st.session_state.edit_chunks     = list(rec.get("gold_chunks", []))
        st.session_state.edit_record_idx = idx


# ─────────────────────────────────────────────────────────────────────────────
# Save helper
# ─────────────────────────────────────────────────────────────────────────────

def save_annotation(idx: int, updates: dict) -> None:
    """Apply updates to record at idx and persist to disk."""
    rec = st.session_state.records[idx]
    rec.update(updates)
    rec["annotation_timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    to_save = [
        r for r in st.session_state.records
        if r.get("annotation_status") != STATUS_UNANNOTATED
    ]
    verified_path = get_verified_path(st.session_state.selected_file)
    save_verified(to_save, verified_path)


# ─────────────────────────────────────────────────────────────────────────────
# Filters
# ─────────────────────────────────────────────────────────────────────────────

def apply_filters(
    records:           list[dict],
    chapter_filter:    str,
    difficulty_filter: str,
    ann_filter:        str,
    gen_filter:        str,
) -> list[int]:
    indices = []
    for i, rec in enumerate(records):
        if chapter_filter != "All":
            if rec.get("chapter") != int(chapter_filter.replace("C", "")):
                continue
        if difficulty_filter != "All":
            if rec.get("difficulty", "") != difficulty_filter.lower():
                continue
        if ann_filter != "All":
            if rec.get("annotation_status") != ann_filter:
                continue
        if gen_filter != "All":
            if rec.get("status", "") != gen_filter:
                continue
        indices.append(i)
    return indices


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar(records: list[dict]) -> tuple[str, str, str, str]:
    with st.sidebar:
        st.title("QAC Review Tool")
        st.markdown("---")

        # ── File selector ─────────────────────────────────────────────────────
        st.subheader("Source file")
        qac_files = list_qac_files()

        if not qac_files:
            st.warning(f"No JSONL files found in:\n`{QAC_DIR}`")
        else:
            file_names = [f.name for f in qac_files]
            current_name = (
                st.session_state.selected_file.name
                if st.session_state.selected_file else None
            )
            default_idx = (
                file_names.index(current_name)
                if current_name in file_names else 0
            )
            chosen_name = st.selectbox(
                "Select QAC file",
                file_names,
                index=default_idx,
                label_visibility="collapsed",
            )
            chosen_path = QAC_DIR / chosen_name

            # Load whenever the selection changes
            if chosen_path != st.session_state.selected_file:
                load_file(chosen_path)
                st.rerun()

            if st.session_state.selected_file:
                verified_path = get_verified_path(st.session_state.selected_file)
                st.caption(f"Output: `{verified_path.name}`")

        st.markdown("---")

        # ── Progress ──────────────────────────────────────────────────────────
        total     = len(records)
        annotated = sum(1 for r in records if r.get("annotation_status") != STATUS_UNANNOTATED)
        approved  = sum(1 for r in records if r.get("annotation_status") in (STATUS_APPROVED, STATUS_EDITED))
        rejected  = sum(1 for r in records if r.get("annotation_status") == STATUS_REJECTED)
        skipped   = annotated - approved - rejected

        st.markdown(f"**Progress: {annotated} / {total}**")
        st.progress(annotated / total if total else 0)
        col1, col2, col3 = st.columns(3)
        col1.metric("✅ Approved", approved)
        col2.metric("❌ Rejected", rejected)
        col3.metric("⏭ Skipped",  skipped)

        st.markdown("---")

        # ── Filters ───────────────────────────────────────────────────────────
        st.subheader("Filters")
        chapters_present = sorted({r.get("chapter", 0) for r in records})
        chapter_options  = ["All"] + [f"C{c:02d}" for c in chapters_present]
        chapter_filter   = st.selectbox("Chapter",    chapter_options)
        diff_filter      = st.selectbox("Difficulty", ["All", "easy", "medium", "hard"])
        ann_filter       = st.selectbox("Annotation", ["All"] + ALL_ANNOTATION_STATUSES)
        gen_filter       = st.selectbox("Gen status", ["All", "passed", "manual_review"])

        st.markdown("---")

        if st.button("⏭ Next unannotated", use_container_width=True):
            filtered = apply_filters(records, chapter_filter, diff_filter,
                                     STATUS_UNANNOTATED, gen_filter)
            if filtered:
                st.session_state.selected = filtered[0]
                st.rerun()
            else:
                st.toast("No unannotated records match current filters.")

    return chapter_filter, diff_filter, ann_filter, gen_filter


# ─────────────────────────────────────────────────────────────────────────────
# Table
# ─────────────────────────────────────────────────────────────────────────────

def render_table(records: list[dict], visible_indices: list[int]) -> None:
    st.subheader(f"Records ({len(visible_indices)} shown)")

    if not visible_indices:
        st.info("No records match the current filters.")
        return

    header = st.columns([0.5, 0.6, 0.8, 4.5, 1.2, 1.4])
    for col, label in zip(header, ["#", "Ch", "Diff", "Question", "Gen", "Annotation"]):
        col.markdown(f"**{label}**")
    st.markdown("<hr style='margin:4px 0'>", unsafe_allow_html=True)

    for i in visible_indices:
        rec        = records[i]
        diff       = rec.get("difficulty", "?")
        diff_col   = DIFFICULTY_COLOURS.get(diff, "#555")
        ann_status = rec.get("annotation_status", STATUS_UNANNOTATED)
        ann_col    = ANNOTATION_COLOURS.get(ann_status, "#555")
        gen_status = rec.get("status", "")
        gen_icon   = "✅" if gen_status == "passed" else "⚠️"
        question   = rec.get("question", "")[:85]
        is_sel     = st.session_state.selected == i
        row_bg     = "background:#1e3a5f;" if is_sel else ""

        cols = st.columns([0.5, 0.6, 0.8, 4.5, 1.2, 1.4])
        cols[0].markdown(f'<div style="{row_bg}padding:2px">{i}</div>', unsafe_allow_html=True)
        cols[1].markdown(f'<div style="{row_bg}padding:2px">C{rec.get("chapter",0):02d}</div>', unsafe_allow_html=True)
        cols[2].markdown(
            f'<div style="{row_bg}padding:2px">'
            f'<span style="color:{diff_col};font-weight:bold">{diff}</span></div>',
            unsafe_allow_html=True,
        )
        cols[3].markdown(
            f'<div style="{row_bg}padding:2px;font-size:13px">{question}</div>',
            unsafe_allow_html=True,
        )
        cols[4].markdown(
            f'<div style="{row_bg}padding:2px">{gen_icon} {gen_status}</div>',
            unsafe_allow_html=True,
        )
        btn_label = "✦ Open" if is_sel else "▶ Open"
        if cols[5].button(btn_label, key=f"sel_{i}"):
            st.session_state.selected = i
            # Reset edit state so it syncs to the new record
            st.session_state.edit_record_idx = None
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Detail panel
# ─────────────────────────────────────────────────────────────────────────────

def render_detail(idx: int, records: list[dict], visible_indices: list[int]) -> None:
    rec = records[idx]

    # Sync dynamic edit lists to this record (runs once on first open)
    sync_edit_state(idx)

    full_md    = st.session_state.full_md
    offsets    = st.session_state.offsets
    wp         = rec.get("window_pages", [0, 0])
    pages_text = extract_pages(full_md, offsets, wp[0], wp[1]) if full_md else ""

    st.markdown("---")

    diff    = rec.get("difficulty", "?")
    ann     = rec.get("annotation_status", STATUS_UNANNOTATED)
    diff_col = DIFFICULTY_COLOURS.get(diff, "#555")
    ann_col  = ANNOTATION_COLOURS.get(ann, "#555")

    st.markdown(
        f'<div style="display:flex;gap:12px;align-items:center;margin-bottom:8px">'
        f'<span style="font-size:18px;font-weight:bold">Record {idx}</span>'
        f'<span style="background:{diff_col};color:white;padding:2px 10px;border-radius:4px">{diff.upper()}</span>'
        f'<span style="background:{ann_col};color:white;padding:2px 10px;border-radius:4px">{ann}</span>'
        f'<span style="color:#888;font-size:13px">'
        f'Chapter {rec.get("chapter")} | Pages {wp[0]}-{wp[1]} | '
        f'Gen: {rec.get("status","?")}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns(2, gap="large")

    # ── LEFT COLUMN ───────────────────────────────────────────────────────────
    with left:
        new_difficulty = st.selectbox(
            "Difficulty",
            ["easy", "medium", "hard"],
            index=["easy", "medium", "hard"].index(rec.get("difficulty", "easy")),
            key=f"diff_{idx}",
        )

        st.markdown("**Question**")
        new_question = st.text_area(
            "Question", value=rec.get("question", ""),
            height=100, key=f"q_{idx}",
            label_visibility="collapsed",
        )

        st.markdown("**Mock Answer**")
        new_mock = st.text_area(
            "Mock Answer", value=rec.get("mock_answer", ""),
            height=150, key=f"mock_{idx}",
            label_visibility="collapsed",
        )

        # ── Rubric (dynamic list backed by session state) ─────────────────────
        st.markdown("**Rubric**")

        # Add / delete buttons mutate session state and rerun immediately
        add_rub = st.button("＋ Add criterion", key=f"add_rub_{idx}")
        if add_rub:
            st.session_state.edit_rubric.append("")
            st.rerun()

        new_rubric: list[str] = []
        delete_rub_idx: int | None = None

        for ri, criterion in enumerate(st.session_state.edit_rubric):
            # Each criterion in a rounded, outlined box
            st.markdown(
                '<div style="border:1px solid #444;border-radius:6px;'
                'padding:6px 8px;margin-bottom:4px">',
                unsafe_allow_html=True,
            )
            rcols = st.columns([10, 1])
            edited = rcols[0].text_area(
                f"Criterion {ri+1}",
                value=criterion,
                height=68,
                key=f"rub_{idx}_{ri}",
                label_visibility="collapsed",
            )
            if rcols[1].button("✕", key=f"del_rub_{idx}_{ri}"):
                delete_rub_idx = ri
            else:
                new_rubric.append(edited)
            st.markdown("</div>", unsafe_allow_html=True)

        if delete_rub_idx is not None:
            st.session_state.edit_rubric.pop(delete_rub_idx)
            st.rerun()
        else:
            # Keep session state in sync with any text edits
            st.session_state.edit_rubric = new_rubric

    # ── RIGHT COLUMN ──────────────────────────────────────────────────────────
    with right:
        st.markdown("**Gold Chunks**")
        st.caption(
            "🟢 found verbatim  |  🔴 not found  |  ⚪ pages unavailable"
        )

        add_chunk = st.button("＋ Add chunk", key=f"add_chunk_{idx}")
        if add_chunk:
            st.session_state.edit_chunks.append("")
            st.rerun()

        new_chunks: list[str] = []
        delete_chunk_idx: int | None = None

        for ci, chunk in enumerate(st.session_state.edit_chunks):
            if pages_text:
                found     = check_chunk(chunk, pages_text)
                indicator = "🟢" if found else "🔴"
            else:
                indicator = "⚪"

            st.markdown(f"**Chunk {ci+1}** {indicator}")
            ccols = st.columns([10, 1])
            edited_chunk = ccols[0].text_area(
                f"Chunk {ci+1}", value=chunk,
                height=80, key=f"chunk_{idx}_{ci}",
                label_visibility="collapsed",
            )
            if ccols[1].button("✕", key=f"del_chunk_{idx}_{ci}"):
                delete_chunk_idx = ci
            else:
                new_chunks.append(edited_chunk)

        if delete_chunk_idx is not None:
            st.session_state.edit_chunks.pop(delete_chunk_idx)
            st.rerun()
        else:
            st.session_state.edit_chunks = new_chunks

        # ── Chunk Relationships ───────────────────────────────────────────────
        with st.expander("Chunk Relationships", expanded=False):
            rels         = rec.get("chunk_relationships", {})
            composites   = rels.get("composites", [])
            substitutes  = rels.get("substitutes", [])
            new_composites  = []
            new_substitutes = []

            st.markdown("**Composites** (all needed together)")
            for gi, group in enumerate(composites):
                edited_group = st.text_area(
                    f"Composite group {gi+1}",
                    value="\n".join(group),
                    height=80, key=f"comp_{idx}_{gi}",
                    help="One sentence per line",
                )
                new_composites.append(edited_group.splitlines())

            st.markdown("**Substitutes** (any one sufficient)")
            for gi, group in enumerate(substitutes):
                edited_group = st.text_area(
                    f"Substitute group {gi+1}",
                    value="\n".join(group),
                    height=80, key=f"subs_{idx}_{gi}",
                    help="One sentence per line",
                )
                new_substitutes.append(edited_group.splitlines())

    # ── Pipeline flag reasons (read-only) ─────────────────────────────────────
    flag_reasons = []
    v = rec.get("verification", {})
    if isinstance(v, dict):
        flag_reasons = v.get("flag_reasons", [])
    if flag_reasons:
        with st.expander(f"⚠️ Pipeline flag reasons ({len(flag_reasons)})", expanded=False):
            for reason in flag_reasons:
                st.markdown(f"- `{reason}`")

    # ── Annotator note ────────────────────────────────────────────────────────
    st.markdown("**Annotator note** (optional)")
    new_note = st.text_input(
        "Note", value=rec.get("annotator_note", ""),
        key=f"note_{idx}", label_visibility="collapsed",
    )

    # ── Action buttons ────────────────────────────────────────────────────────
    st.markdown("---")
    bcols = st.columns(5)

    def collect_edits() -> dict:
        """Build a diff of only the fields that actually changed."""
        edited: dict = {}
        if new_question   != rec.get("question"):     edited["question"]    = new_question
        if new_mock       != rec.get("mock_answer"):  edited["mock_answer"] = new_mock
        if new_difficulty != rec.get("difficulty"):   edited["difficulty"]  = new_difficulty
        if st.session_state.edit_rubric != rec.get("rubric"):
            edited["rubric"] = list(st.session_state.edit_rubric)
        if st.session_state.edit_chunks != rec.get("gold_chunks"):
            edited["gold_chunks"] = list(st.session_state.edit_chunks)
        new_rels = {"composites": new_composites, "substitutes": new_substitutes}
        if new_rels != rec.get("chunk_relationships"):
            edited["chunk_relationships"] = new_rels
        return edited

    def apply_edits(edited: dict) -> None:
        for field in ("question", "mock_answer", "difficulty",
                      "rubric", "gold_chunks", "chunk_relationships"):
            if field in edited:
                rec[field] = edited[field]

    if bcols[0].button("✅ Approve", key=f"approve_{idx}", use_container_width=True):
        edited = collect_edits()
        apply_edits(edited)
        save_annotation(idx, {
            "annotation_status": STATUS_EDITED if edited else STATUS_APPROVED,
            "annotator_note":    new_note,
            "edited_fields":     edited,
        })
        st.toast(f"✅ {'Edited & approved' if edited else 'Approved'}")
        st.rerun()

    if bcols[1].button("✏️ Edit & Approve", key=f"edit_{idx}", use_container_width=True):
        edited = collect_edits()
        apply_edits(edited)
        save_annotation(idx, {
            "annotation_status": STATUS_EDITED,
            "annotator_note":    new_note,
            "edited_fields":     edited,
        })
        st.toast("✏️ Edited & approved")
        st.rerun()

    if bcols[2].button("❌ Reject", key=f"reject_{idx}", use_container_width=True):
        save_annotation(idx, {
            "annotation_status": STATUS_REJECTED,
            "annotator_note":    new_note,
            "edited_fields":     {},
        })
        st.toast("❌ Rejected")
        st.rerun()

    if bcols[3].button("⏭ Skip", key=f"skip_{idx}", use_container_width=True):
        save_annotation(idx, {
            "annotation_status": STATUS_SKIPPED,
            "annotator_note":    new_note,
            "edited_fields":     {},
        })
        unannotated = [
            i for i, r in enumerate(records)
            if r.get("annotation_status") == STATUS_UNANNOTATED and i != idx
        ]
        if unannotated:
            st.session_state.selected        = unannotated[0]
            st.session_state.edit_record_idx = None
        st.toast("⏭ Skipped")
        st.rerun()

    if bcols[4].button("→ Next", key=f"next_{idx}", use_container_width=True):
        current_pos = visible_indices.index(idx) if idx in visible_indices else -1
        next_pos = current_pos + 1
        if next_pos < len(visible_indices):
            st.session_state.selected = visible_indices[next_pos]
            st.session_state.edit_record_idx = None
            st.rerun()
        else:
            st.toast("Already at the last record in current filter.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="QAC Review Tool",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown("""
    <style>
        .block-container { padding-top: 1rem; }
        .stButton button  { font-size: 13px; }
        div[data-testid="stTextArea"] textarea {
            font-size: 13px;
            font-family: monospace;
        }
    </style>
    """, unsafe_allow_html=True)

    init_state()

    qac_files = list_qac_files()
    if not qac_files:
        st.error(
            f"No JSONL files found in `{QAC_DIR}`.\n\n"
            "Run the generation pipeline first:\n"
            "```\n"
            "python src/llm_benchmark_generation/main.py generate "
            "--chapters all --windows first\n"
            "```"
        )
        return

    # Auto-load the first file if nothing is selected yet
    if st.session_state.selected_file is None:
        load_file(qac_files[0])
        st.rerun()

    records = st.session_state.records

    chapter_f, diff_f, ann_f, gen_f = render_sidebar(records)
    visible = apply_filters(records, chapter_f, diff_f, ann_f, gen_f)

    render_table(records, visible)

    if st.session_state.selected is not None:
        idx = st.session_state.selected
        if 0 <= idx < len(records):
            render_detail(idx, records, visible)
    else:
        st.info(
            "Select a record from the table above, "
            "or click **⏭ Next unannotated** in the sidebar."
        )


if __name__ == "__main__":
    main()