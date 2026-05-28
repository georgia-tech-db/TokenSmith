"""
src/benchmark_eval/results_viewer_external.py

Streamlit viewer for external benchmark results produced by
run_external_benchmark.py.

Run from the TokenSmith project root:
    streamlit run src/benchmark_eval/results_viewer_external.py

Reads:
    benchmark_results/{run_label}/full_results.json
"""

import json
import pathlib
import re
import sys

import streamlit as st

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

RESULTS_ROOT = _PROJECT_ROOT / "benchmark_results"

# ─────────────────────────────────────────────────────────────────────────────
# Page config + CSS
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="TokenSmith — External Benchmark Viewer",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.stApp { background-color: #0d1117; color: #e2e8f0; }

section[data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #21262d;
}
section[data-testid="stSidebar"] * { color: #c9d1d9 !important; }
section[data-testid="stSidebar"] label {
    font-size: 0.78rem !important;
    color: #8b949e !important;
}

.block-container { padding-top: 3.5rem; max-width: 1500px; }

.section-hdr {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #30363d;
    border-bottom: 1px solid #21262d;
    padding-bottom: 5px;
    margin: 16px 0 10px 0;
}

/* Score colours */
.s-pos  { color: #3fb950 !important; }
.s-zero { color: #d29922 !important; }
.s-neg  { color: #f85149 !important; }
.s-neu  { color: #58a6ff !important; }

/* Summary scorecards */
.score-grid {
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 8px;
    margin-bottom: 16px;
}
.score-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 10px 14px;
    text-align: center;
}
.score-card .sc-label {
    font-size: 0.63rem;
    color: #6e7681;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 4px;
}
.score-card .sc-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.3rem;
    font-weight: 600;
}

/* Nav bar */
.nav-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 9px 16px;
    margin-bottom: 14px;
}

/* Question box */
.q-box {
    background: #161b22;
    border: 1px solid #21262d;
    border-left: 3px solid #58a6ff;
    border-radius: 6px;
    padding: 13px 16px;
    font-size: 1.0rem;
    font-weight: 500;
    line-height: 1.6;
    color: #e6edf3;
    margin-bottom: 12px;
}

/* Rubric items */
.rub-item {
    display: flex;
    align-items: flex-start;
    gap: 8px;
    padding: 7px 0;
    border-bottom: 1px solid #21262d;
    font-size: 0.84rem;
    line-height: 1.5;
    color: #c9d1d9;
}
.rub-item:last-child { border-bottom: none; }

/* Chunk cards */
.chunk-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-left: 3px solid;
    border-radius: 6px;
    padding: 11px 14px;
    margin-bottom: 8px;
    font-size: 0.84rem;
    line-height: 1.65;
    color: #c9d1d9;
}
.cc-pos { border-left-color: #3fb950; }
.cc-neg { border-left-color: #f85149; }
.cc-unc { border-left-color: #d29922; }
.cc-neu { border-left-color: #30363d; }

.chunk-meta {
    display: flex;
    gap: 14px;
    margin-bottom: 7px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: #6e7681;
    flex-wrap: wrap;
}
.chunk-meta .rank { color: #58a6ff; font-weight: 600; }

.chunk-reason {
    margin-top: 7px;
    font-size: 0.78rem;
    color: #6e7681;
    font-style: italic;
    border-top: 1px solid #21262d;
    padding-top: 5px;
}

/* Answer + verdict boxes */
.answer-box {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 14px 16px;
    font-size: 0.85rem;
    line-height: 1.75;
    color: #e6edf3;
    white-space: pre-wrap;
}
.verdict-box {
    background: #161b22;
    border-left: 3px solid #58a6ff;
    border-radius: 0 6px 6px 0;
    padding: 11px 14px;
    font-size: 0.83rem;
    font-style: italic;
    color: #8b949e;
    line-height: 1.65;
    margin-top: 8px;
}

/* Inner cards */
.inner-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 12px 16px;
}

/* Optional rubric section label */
.opt-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: #6e7681;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 10px 0 4px 0;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def list_run_dirs() -> list:
    """Return run directories that contain full_results.json."""
    if not RESULTS_ROOT.exists():
        return []
    valid = []
    for d in RESULTS_ROOT.iterdir():
        if d.is_dir() and (d / "full_results.json").exists():
            valid.append(d)
    return sorted(valid, key=lambda d: d.stat().st_mtime, reverse=True)


@st.cache_data
def load_run(run_dir: str) -> list:
    path = pathlib.Path(run_dir) / "full_results.json"
    with open(path, encoding="utf-8") as f:
        records = json.load(f)

    # Pre-compute summary fields for filtering / table display
    for r in records:
        s = r.get("summary", {})
        j = r.get("judgements", {})

        r["_must_met"]   = s.get("must_rubric_met_count", 0)
        r["_must_total"] = s.get("must_rubric_total", 0)
        r["_opt_met"]    = s.get("opt_rubric_met_count", 0)
        r["_opt_total"]  = s.get("opt_rubric_total", 0)
        r["_score_ref"]  = s.get("correctness_with_ref")
        r["_score_noref"]= s.get("correctness_no_ref")
        r["_faith"]      = s.get("faithfulness_verdict", "")
        r["_chunk_rel"]  = s.get("chunk_relevance_rate")

        rel_items = (
            j.get("chunk_relevance", {}).get("group")
            or j.get("chunk_relevance", {}).get("individual")
            or []
        )
        r["_rel_by_rank"] = {rr.get("rank"): rr for rr in rel_items}

    return records


# ─────────────────────────────────────────────────────────────────────────────
# UI helpers
# ─────────────────────────────────────────────────────────────────────────────

SCORE_COLOUR = {1: "#3fb950", 0: "#d29922", -1: "#f85149"}
SCORE_LABEL  = {1: "✅ +1",    0: "⚠️ 0",    -1: "❌ −1"}
FAITH_COLOUR = {
    "faithful":           "#3fb950",
    "partially_faithful": "#d29922",
    "unfaithful":         "#f85149",
    "uncertain":          "#6e7681",
}
FAITH_SHORT = {
    "faithful": "✅", "partially_faithful": "⚠️",
    "unfaithful": "❌", "uncertain": "❓",
}
VERDICT_COLOUR = {
    "met": "#3fb950", "partial": "#d29922", "not_met": "#f85149",
    "relevant": "#3fb950", "not_relevant": "#f85149", "uncertain": "#d29922",
}
VERDICT_ICON = {
    "met": "✅", "partial": "⚠️", "not_met": "❌",
    "relevant": "✅", "not_relevant": "❌", "uncertain": "❓",
}


def badge(text: str, colour: str) -> str:
    return (
        f'<span style="background:{colour};color:#fff;padding:2px 8px;'
        f'border-radius:4px;font-size:12px;font-weight:600;white-space:nowrap">'
        f'{text}</span>'
    )


def mono(text: str, colour: str = "#58a6ff", size: str = "0.72rem") -> str:
    return (
        f'<span style="font-family:IBM Plex Mono,monospace;'
        f'font-size:{size};color:{colour}">{text}</span>'
    )


def section(label: str) -> None:
    st.markdown(f'<div class="section-hdr">{label}</div>', unsafe_allow_html=True)


def inner_card(html: str) -> None:
    st.markdown(f'<div class="inner-card">{html}</div>', unsafe_allow_html=True)


def rubric_rows(items: list, is_optional: bool = False) -> str:
    html = ""
    for rr in items:
        verdict = rr.get("verdict", "")
        reason  = rr.get("reason", "")
        vc      = VERDICT_COLOUR.get(verdict, "#6e7681")
        vi      = VERDICT_ICON.get(verdict, "❓")
        vl      = verdict.replace("_", " ").title()

        verdict_span = (
            f'<span style="font-family:IBM Plex Mono,monospace;'
            f'font-size:0.65rem;color:{vc};font-weight:600;margin-left:6px">'
            f'{vi} {vl}</span>'
        ) if verdict else ""

        optional_tag = (
            f'<span style="font-family:IBM Plex Mono,monospace;'
            f'font-size:0.60rem;color:#6e7681;margin-left:6px;'
            f'border:1px solid #30363d;padding:0 4px;border-radius:3px">optional</span>'
            if is_optional else ""
        )

        reason_html = ""
        if reason:
            reason_html = (
                f'<div style="font-size:0.75rem;color:#6e7681;'
                f'font-style:italic;margin-top:3px">{reason}</div>'
            )

        html += (
            f'<div class="rub-item">'
            f'<div style="flex:1">'
            f'<div>{rr["criterion"]}{verdict_span}{optional_tag}</div>'
            f'{reason_html}'
            f'</div>'
            f'</div>'
        )
    return html


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        '<span style="font-family:IBM Plex Mono,monospace;font-size:1.0rem;'
        'font-weight:600;color:#58a6ff">⬡ External Benchmark Viewer</span>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    run_dirs = list_run_dirs()
    if not run_dirs:
        st.error(f"No `full_results.json` found under:\n`{RESULTS_ROOT}`")
        st.stop()

    run_names    = [d.name for d in run_dirs]
    selected_run = st.selectbox("Run", run_names, label_visibility="collapsed")
    run_path     = RESULTS_ROOT / selected_run

    records = load_run(str(run_path))
    if not records:
        st.error("No records in this run.")
        st.stop()

    n = len(records)

    # Judge backend info
    judge_desc = records[0].get("judge_backend", "unknown") if records else "unknown"
    st.caption(f"Judge: `{judge_desc}`")
    st.caption(f"{n} questions")

    st.markdown("---")
    st.markdown(
        '<span style="font-family:IBM Plex Mono,monospace;font-size:0.72rem;'
        'color:#6e7681;letter-spacing:0.08em">FILTERS</span>',
        unsafe_allow_html=True,
    )

    score_filter = st.multiselect(
        "Correctness (with ref)",
        options=[1, 0, -1],
        default=[1, 0, -1],
        format_func=lambda x: {1: "✅ Correct", 0: "⚠️ Partial", -1: "❌ Wrong"}[x],
    )

    must_cov_filter = st.selectbox(
        "Must rubric coverage",
        ["All", "Full (100%)", "Partial", "None (0%)"],
    )

    faith_filter = st.selectbox(
        "Faithfulness",
        ["All", "Faithful", "Partially faithful", "Unfaithful", "Uncertain"],
    )

    st.markdown("---")
    if st.button("↺ Reload data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Apply filters
# ─────────────────────────────────────────────────────────────────────────────

FAITH_FILTER_MAP = {
    "Faithful": "faithful", "Partially faithful": "partially_faithful",
    "Unfaithful": "unfaithful", "Uncertain": "uncertain",
}

visible = []
for i, r in enumerate(records):
    if r["_score_ref"] not in score_filter:
        continue
    if must_cov_filter != "All":
        nm, nt = r["_must_met"], r["_must_total"]
        if must_cov_filter == "Full (100%)" and nm < nt:           continue
        if must_cov_filter == "None (0%)"   and nm > 0:            continue
        if must_cov_filter == "Partial"     and not (0 < nm < nt): continue
    if faith_filter != "All":
        if r["_faith"] != FAITH_FILTER_MAP.get(faith_filter, ""):
            continue
    visible.append(i)

# ─────────────────────────────────────────────────────────────────────────────
# Navigation state
# ─────────────────────────────────────────────────────────────────────────────

if "nav_idx" not in st.session_state:
    st.session_state.nav_idx = 0

filter_sig = str((selected_run, tuple(score_filter), must_cov_filter, faith_filter))
if st.session_state.get("_fsig") != filter_sig:
    st.session_state.nav_idx = 0
    st.session_state["_fsig"] = filter_sig

if not visible:
    st.markdown(
        '<div style="text-align:center;padding:80px;font-family:IBM Plex Mono,'
        'monospace;color:#30363d;font-size:1.0rem">No records match current filters</div>',
        unsafe_allow_html=True,
    )
    st.stop()

nav = min(st.session_state.nav_idx, len(visible) - 1)
rec = records[visible[nav]]
n_vis = len(visible)


# ─────────────────────────────────────────────────────────────────────────────
# Nav bar
# ─────────────────────────────────────────────────────────────────────────────

c_prev, c_info, c_next = st.columns([1, 5, 1])
with c_prev:
    if st.button("← Prev", disabled=(nav == 0), use_container_width=True):
        st.session_state.nav_idx = max(0, nav - 1)
        st.rerun()
with c_info:
    qid   = rec.get("id", "?")
    sc_w  = rec["_score_ref"]
    sc_wc = SCORE_COLOUR.get(sc_w, "#6e7681")
    sc_wl = SCORE_LABEL.get(sc_w, "—")
    fv    = rec["_faith"]
    fvc   = FAITH_COLOUR.get(fv, "#6e7681")
    fvs   = FAITH_SHORT.get(fv, "—")
    st.markdown(
        f'<div class="nav-bar">'
        + mono(f"Q {qid}", "#58a6ff", "0.9rem")
        + f'<span style="display:flex;gap:12px;align-items:center">'
        + f'<span style="font-family:IBM Plex Mono,monospace;font-size:0.78rem;'
        + f'color:{sc_wc}">{sc_wl}</span>'
        + f'<span style="font-family:IBM Plex Mono,monospace;font-size:0.78rem;'
        + f'color:{fvc}">{fvs} {fv.replace("_"," ").title()}</span>'
        + f'</span>'
        + f'<span style="font-family:IBM Plex Mono,monospace;font-size:0.72rem;'
        + f'color:#6e7681">{nav+1} / {n_vis}'
        + (f"  ({n} total)" if n_vis != n else "")
        + f'</span>'
        + f'</div>',
        unsafe_allow_html=True,
    )
with c_next:
    if st.button("Next →", disabled=(nav == n_vis - 1), use_container_width=True):
        st.session_state.nav_idx = min(n_vis - 1, nav + 1)
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Question
# ─────────────────────────────────────────────────────────────────────────────

item = rec.get("benchmark_item", {})
j    = rec.get("judgements", {})
s    = rec.get("summary", {})

st.markdown(
    f'<div class="q-box">'
    + mono(f"Question {item.get('id','?')}", "#6e7681", "0.68rem")
    + f'<div style="margin-top:4px">{item.get("question","")}</div>'
    + f'</div>',
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Summary scorecards
# ─────────────────────────────────────────────────────────────────────────────

nm, nt  = rec["_must_met"], rec["_must_total"]
om, ot  = rec["_opt_met"],  rec["_opt_total"]
sc_w    = rec["_score_ref"]
sc_n    = rec["_score_noref"]
fv      = rec["_faith"]
cr      = rec["_chunk_rel"]

must_c  = "#3fb950" if nm == nt else ("#d29922" if nm > 0 else "#f85149")
opt_c   = "#3fb950" if (ot == 0 or om == ot) else ("#d29922" if om > 0 else "#6e7681")
sc_wc   = SCORE_COLOUR.get(sc_w, "#6e7681")
sc_nc   = SCORE_COLOUR.get(sc_n, "#6e7681")
fvc     = FAITH_COLOUR.get(fv, "#6e7681")
cr_c    = "#3fb950" if (cr or 0) >= 0.6 else ("#d29922" if (cr or 0) >= 0.3 else "#f85149")

st.markdown(
    f'<div class="score-grid">'

    f'<div class="score-card">'
    f'<div class="sc-label">Must Rubric</div>'
    f'<div class="sc-value" style="color:{must_c}">{nm}/{nt}</div>'
    f'</div>'

    f'<div class="score-card">'
    f'<div class="sc-label">Optional Rubric</div>'
    f'<div class="sc-value" style="color:{opt_c}">{om}/{ot}</div>'
    f'</div>'

    f'<div class="score-card">'
    f'<div class="sc-label">Correct (ref)</div>'
    f'<div class="sc-value" style="color:{sc_wc}">{SCORE_LABEL.get(sc_w,"—")}</div>'
    f'</div>'

    f'<div class="score-card">'
    f'<div class="sc-label">Correct (no ref)</div>'
    f'<div class="sc-value" style="color:{sc_nc}">{SCORE_LABEL.get(sc_n,"—")}</div>'
    f'</div>'

    f'<div class="score-card">'
    f'<div class="sc-label">Faithfulness</div>'
    f'<div class="sc-value" style="color:{fvc}">'
    f'{FAITH_SHORT.get(fv,"—")} {fv.replace("_"," ").title() if fv else "—"}'
    f'</div>'
    f'</div>'

    f'<div class="score-card">'
    f'<div class="sc-label">Chunk Relevance</div>'
    f'<div class="sc-value" style="color:{cr_c}">'
    f'{f"{cr*100:.0f}%" if cr is not None else "—"}'
    f'</div>'
    f'</div>'

    f'</div>',
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Main content — two columns
# ─────────────────────────────────────────────────────────────────────────────

answer_clean = re.sub(r"<<<[A-Z_]+>>>", "", rec.get("ts_answer", "")).strip()
corr_w  = j.get("answer_correctness", {}).get("with_reference", {})
corr_n  = j.get("answer_correctness", {}).get("without_reference", {})
sc_w2   = corr_w.get("score")
sc_n2   = corr_n.get("score")
sc_w2c  = SCORE_COLOUR.get(sc_w2, "#58a6ff")
sc_n2c  = SCORE_COLOUR.get(sc_n2, "#58a6ff")
sc_w2l  = SCORE_LABEL.get(sc_w2, "—")
sc_n2l  = SCORE_LABEL.get(sc_n2, "—")
exp_w   = corr_w.get("explanation", "")
exp_n   = corr_n.get("explanation", "")

must_res = j.get("must_rubric_satisfaction", [])
opt_res  = j.get("optional_rubric_satisfaction", [])

faith    = j.get("faithfulness", {})
fv2      = faith.get("verdict", "")
fv2c     = FAITH_COLOUR.get(fv2, "#6e7681")
fv2l     = fv2.replace("_", " ").title() if fv2 else ""
fv2s     = FAITH_SHORT.get(fv2, "")
fexp     = faith.get("explanation", "")
unsup    = faith.get("unsupported_claims", [])
faith_skipped = fv2 in ("", "skipped", "uncertain") and not fexp and not unsup

chunks      = rec.get("retrieved_chunks", [])
rel_by_rank = rec.get("_rel_by_rank", {})

col_l, col_r = st.columns([5, 4], gap="large")

# ─────────────────────────────── LEFT ────────────────────────────────────────
with col_l:

    # ── Generated Answer ──────────────────────────────────────────────────────
    with st.expander("🤖 Generated Answer", expanded=True):
        st.markdown(
            f'<div style="display:flex;gap:8px;margin-bottom:10px">'
            + badge(f"With ref: {sc_w2l}", sc_w2c)
            + " "
            + badge(f"No ref: {sc_n2l}", sc_n2c)
            + f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="answer-box">{answer_clean}</div>',
            unsafe_allow_html=True,
        )

    # ── Mock Answer ───────────────────────────────────────────────────────────
    with st.expander("📖 Mock / Reference Answer", expanded=False):
        st.markdown(
            f'<div class="answer-box">{item.get("answer", "")}</div>',
            unsafe_allow_html=True,
        )

    # ── Judge — With Reference ────────────────────────────────────────────────
    with st.expander(f"⚖️ Judge Score — With Reference  {sc_w2l}", expanded=True):
        if exp_w:
            st.markdown(
                f'<div class="verdict-box">'
                + f'<span style="font-family:IBM Plex Mono,monospace;font-size:0.72rem;'
                + f'font-weight:600;color:{sc_w2c};font-style:normal">{sc_w2l}</span>'
                + f'<br>{exp_w}'
                + f'</div>',
                unsafe_allow_html=True,
            )
            strengths  = corr_w.get("strengths", [])
            weaknesses = corr_w.get("weaknesses", [])
            if strengths:
                st.markdown("**Strengths:**")
                for s_item in strengths:
                    st.markdown(f"- {s_item}")
            if weaknesses:
                st.markdown("**Weaknesses:**")
                for w_item in weaknesses:
                    st.markdown(f"- {w_item}")
        else:
            st.caption("No explanation available.")

    # ── Judge — Without Reference ─────────────────────────────────────────────
    with st.expander(f"⚖️ Judge Score — Without Reference  {sc_n2l}", expanded=False):
        if exp_n:
            st.markdown(
                f'<div class="verdict-box">'
                + f'<span style="font-family:IBM Plex Mono,monospace;font-size:0.72rem;'
                + f'font-weight:600;color:{sc_n2c};font-style:normal">{sc_n2l}</span>'
                + f'<br>{exp_n}'
                + f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.caption("No explanation available.")

# ─────────────────────────────── RIGHT ───────────────────────────────────────
with col_r:

    # ── Must Rubric ───────────────────────────────────────────────────────────
    nm_label = f"{nm}/{nt}" if nt else "—"
    must_c_label = "#3fb950" if nm == nt else ("#d29922" if nm > 0 else "#f85149")
    with st.expander(
        f"📋 Must Rubric  "
        f"{'✅' if nm == nt else ('⚠️' if nm > 0 else '❌')} {nm_label}",
        expanded=True,
    ):
        if must_res:
            inner_card(rubric_rows(must_res, is_optional=False))
        else:
            st.caption("No must rubric results.")

    # ── Optional Rubric ───────────────────────────────────────────────────────
    if opt_res:
        om_label = f"{om}/{ot}" if ot else "—"
        with st.expander(
            f"💡 Optional Rubric  "
            f"{'✅' if om == ot else ('⚠️' if om > 0 else '➖')} {om_label}",
            expanded=False,
        ):
            inner_card(rubric_rows(opt_res, is_optional=True))

    # ── Retrieved Chunks ──────────────────────────────────────────────────────
    n_rel = sum(
        1 for rr in rel_by_rank.values() if rr.get("verdict") == "relevant"
    )
    with st.expander(
        f"📦 Retrieved Chunks  ({len(chunks)} chunks, {n_rel} relevant)",
        expanded=False,
    ):
        for chunk in chunks:
            rank    = chunk.get("rank", "?")
            rr      = rel_by_rank.get(rank, {})
            verdict = rr.get("verdict", "")
            reason  = rr.get("reason", "")
            vc      = VERDICT_COLOUR.get(verdict, "#30363d")
            vi      = VERDICT_ICON.get(verdict, "")
            vl      = verdict.replace("_", " ").title()
            faiss   = chunk.get("faiss_score", 0)
            bm25    = chunk.get("bm25_score", 0)
            cc      = {"relevant": "cc-pos", "not_relevant": "cc-neg",
                       "uncertain": "cc-unc"}.get(verdict, "cc-neu")
            content_escaped = (
                chunk.get("content", "")
                .replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            )
            verdict_row = ""
            if verdict:
                verdict_row = (
                    f'<div style="margin-bottom:5px">'
                    f'<span style="font-family:IBM Plex Mono,monospace;'
                    f'font-size:0.65rem;color:{vc};font-weight:600">'
                    f'{vi} {vl}</span></div>'
                )
            reason_row = f'<div class="chunk-reason">💬 {reason}</div>' if reason else ""
            st.markdown(
                f'<div class="chunk-card {cc}">'
                f'<div class="chunk-meta">'
                f'<span class="rank">Rank {rank}</span>'
                f'<span>FAISS {faiss:.2e}</span>'
                f'<span>BM25 {bm25:.1f}</span>'
                f'</div>'
                + verdict_row
                + f'<div style="font-size:0.83rem;line-height:1.65;'
                f'max-height:200px;overflow-y:auto">{content_escaped}</div>'
                + reason_row
                + f'</div>',
                unsafe_allow_html=True,
            )

    # ── Faithfulness ──────────────────────────────────────────────────────────
    if not faith_skipped:
        with st.expander(
            f"🔒 Faithfulness  {fv2s} {fv2l}",
            expanded=False,
        ):
            if fexp:
                st.markdown(
                    f'<div class="verdict-box">'
                    + f'<span style="font-family:IBM Plex Mono,monospace;'
                    + f'font-size:0.72rem;font-weight:600;color:{fv2c};'
                    + f'font-style:normal">{fv2s} {fv2l}</span>'
                    + f'<br>{fexp}'
                    + f'</div>',
                    unsafe_allow_html=True,
                )
            if unsup:
                st.markdown(f"**⚠️ Unsupported claims ({len(unsup)}):**")
                for claim in unsup:
                    st.markdown(f"- {claim}")


# ─────────────────────────────────────────────────────────────────────────────
# Question list (collapsed)
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("")
with st.expander(f"📋 All {n_vis} questions — click to browse", expanded=False):

    hcols = st.columns([0.6, 4.0, 0.9, 0.9, 0.8, 0.8])
    for col, lbl in zip(hcols, ["ID", "Question", "Must", "Correct", "Faith", ""]):
        col.markdown(
            f'<span style="font-family:IBM Plex Mono,monospace;'
            f'font-size:0.65rem;color:#30363d;'
            f'text-transform:uppercase;letter-spacing:0.1em">{lbl}</span>',
            unsafe_allow_html=True,
        )
    st.markdown(
        '<hr style="margin:4px 0;border-color:#21262d">',
        unsafe_allow_html=True,
    )

    for list_pos, vi in enumerate(visible):
        r       = records[vi]
        is_sel  = (vi == visible[nav])
        bg      = "background:#161b22;" if is_sel else ""
        nm2, nt2 = r["_must_met"], r["_must_total"]
        must_c2  = "#3fb950" if nm2 == nt2 else ("#d29922" if nm2 > 0 else "#f85149")
        sc2      = r["_score_ref"]
        sc2c     = SCORE_COLOUR.get(sc2, "#6e7681")
        sc2s     = {1: "✅ 1", 0: "⚠️ 0", -1: "❌ -1"}.get(sc2, "—")
        fv3      = r["_faith"]
        fv3c     = FAITH_COLOUR.get(fv3, "#6e7681")
        fv3s     = FAITH_SHORT.get(fv3, "—")

        qtext    = r.get("benchmark_item", {}).get("question", "")
        q_trunc  = qtext[:70] + ("…" if len(qtext) > 70 else "")
        qid2     = r.get("benchmark_item", {}).get("id", r.get("id", "?"))

        row = st.columns([0.6, 4.0, 0.9, 0.9, 0.8, 0.8])
        row[0].markdown(
            f'<div style="{bg}padding:2px;font-family:IBM Plex Mono,monospace;'
            f'font-size:11px;color:#58a6ff">{qid2}</div>',
            unsafe_allow_html=True)
        row[1].markdown(
            f'<div style="{bg}padding:2px;font-size:12px">{q_trunc}</div>',
            unsafe_allow_html=True)
        row[2].markdown(
            f'<div style="{bg}padding:2px">'
            f'<span style="color:{must_c2};font-size:12px;font-weight:600">'
            f'{nm2}/{nt2}</span></div>',
            unsafe_allow_html=True)
        row[3].markdown(
            f'<div style="{bg}padding:2px">'
            f'<span style="color:{sc2c};font-size:12px;font-weight:600">'
            f'{sc2s}</span></div>',
            unsafe_allow_html=True)
        row[4].markdown(
            f'<div style="{bg}padding:2px">'
            f'<span style="color:{fv3c};font-size:12px">{fv3s}</span></div>',
            unsafe_allow_html=True)

        if row[5].button("▶", key=f"jump_{vi}"):
            st.session_state.nav_idx = list_pos
            st.rerun()