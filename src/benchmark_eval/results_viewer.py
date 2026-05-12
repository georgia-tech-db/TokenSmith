"""
src/benchmark_eval/results_viewer.py

Streamlit app for viewing benchmark evaluation results.

Run from the project root:
    streamlit run src/benchmark_eval/results_viewer.py

Reads from benchmark_results/{run_label}/:
    - raw_results.jsonl
    - judge_results.jsonl
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
# Substring verification — same 4-tier logic used throughout the pipeline
# ─────────────────────────────────────────────────────────────────────────────

def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _clean(text: str) -> str:
    text = re.sub(r"^\s*[\.,!?;:\-_]+\s*$", " ", text, flags=re.MULTILINE)
    text = re.sub(r"---\s*Page\s+\d+\s*---", " ", text)
    text = re.sub(r"\bPage\s+\d+\b", " ", text)
    return _norm(text)


def gold_chunk_found(gold: str, retrieved_chunks: list) -> tuple:
    """Return (found: bool, rank: int|None) for a gold chunk against retrieved chunks."""
    ng = _norm(gold)
    sg = ng.rstrip(".,;:!?")
    for item in retrieved_chunks:
        ct = item.get("content", "")
        nt = _norm(ct)
        cl = _clean(ct)
        if ng in nt or ng in cl or sg in nt or sg in cl:
            return True, item.get("rank")
    return False, None


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def list_run_dirs() -> list:
    if not RESULTS_ROOT.exists():
        return []
    return sorted(
        [d for d in RESULTS_ROOT.iterdir() if d.is_dir()],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )


def load_jsonl(path: pathlib.Path) -> list:
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


@st.cache_data
def load_run(run_dir: str) -> list:
    """Load and join raw_results + judge_results. Returns merged list."""
    path   = pathlib.Path(run_dir)
    raw    = load_jsonl(path / "raw_results.jsonl")
    judges = load_jsonl(path / "judge_results.jsonl")

    judge_idx = {}
    for j in judges:
        rid = j.get("record_id", "")
        if rid:
            judge_idx[rid] = j.get("judgements", {})

    merged = []
    for i, r in enumerate(raw):
        qac = r.get("qac", {})
        rid = qac.get("record_id", f"q{i:04d}")
        retrieved = r.get("retrieved_chunks", [])

        gold_status = []
        for gc in qac.get("gold_chunks", []):
            found, rank = gold_chunk_found(gc, retrieved)
            gold_status.append({"chunk": gc, "found": found, "rank": rank})

        merged.append({
            "record_id":         rid,
            "chapter":           qac.get("chapter"),
            "window_pages":      qac.get("window_pages", []),
            "difficulty":        qac.get("difficulty", "?"),
            "question":          qac.get("question", ""),
            "ts_answer":         r.get("ts_answer", ""),
            "retrieved_chunks":  retrieved,
            "gold_chunks":       qac.get("gold_chunks", []),
            "gold_chunk_flags":  qac.get("gold_chunk_flags", []),
            "gold_status":       gold_status,
            "rubric":            qac.get("rubric", []),
            "rubric_flags":      qac.get("rubric_flags", []),
            "mock_answer":       qac.get("mock_answer", ""),
            "generation_status": qac.get("status", ""),
            "error":             r.get("error"),
            "judgements":        judge_idx.get(rid, {}),
        })

    return merged


# ─────────────────────────────────────────────────────────────────────────────
# UI helpers — no nested f-strings anywhere
# ─────────────────────────────────────────────────────────────────────────────

DIFF_COLOUR  = {"easy": "#2d6a4f", "medium": "#b5770c", "hard": "#a4262c"}
SCORE_COLOUR = {1: "#2d6a4f", 0: "#b5770c", -1: "#a4262c"}
SCORE_LABEL  = {1: "✅ Fully correct", 0: "⚠️ Partially correct", -1: "❌ Incorrect"}
VERDICT_COLOUR = {
    "met":                "#2d6a4f",
    "partial":            "#b5770c",
    "not_met":            "#a4262c",
    "relevant":           "#2d6a4f",
    "not_relevant":       "#a4262c",
    "uncertain":          "#555",
    "faithful":           "#2d6a4f",
    "partially_faithful": "#b5770c",
    "unfaithful":         "#a4262c",
}
VERDICT_ICON = {
    "met": "✅", "partial": "⚠️", "not_met": "❌",
    "relevant": "✅", "not_relevant": "❌", "uncertain": "❓",
    "faithful": "✅", "partially_faithful": "⚠️", "unfaithful": "❌",
}


def badge(text: str, colour: str) -> str:
    return (
        f'<span style="background:{colour};color:#fff;padding:2px 8px;'
        f'border-radius:4px;font-size:12px;font-weight:600;white-space:nowrap">'
        f'{text}</span>'
    )


def card(content_html: str, border_colour: str = "#30363d",
         bg: str = "#0d1117", extra_style: str = "") -> str:
    return (
        f'<div style="background:{bg};border:1px solid #30363d;'
        f'border-left:3px solid {border_colour};'
        f'border-radius:6px;padding:10px 14px;margin-bottom:6px;{extra_style}">'
        f'{content_html}'
        f'</div>'
    )


def mono_block(text: str) -> str:
    escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return (
        f'<div style="font-family:monospace;font-size:12px;line-height:1.6;'
        f'color:#e6edf3;white-space:pre-wrap">{escaped}</div>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# Detail renderer
# ─────────────────────────────────────────────────────────────────────────────

def render_record(rec: dict) -> None:
    j    = rec["judgements"]
    diff = rec["difficulty"]
    dc   = DIFF_COLOUR.get(diff, "#666")

    gen_st  = rec["generation_status"]
    gen_col = "#2d6a4f" if gen_st == "passed" else "#a4262c"

    wp = rec["window_pages"]
    wp_str = f"{wp[0]}-{wp[1]}" if wp else "?"

    # ── Record header ─────────────────────────────────────────────────────────
    header_html = (
        f'<div style="display:flex;gap:8px;align-items:center;'
        f'flex-wrap:wrap;margin-bottom:12px">'
        + badge(diff.upper(), dc) + " "
        + badge(f"Gen: {gen_st}", gen_col) + " "
        + f'<span style="color:#888;font-size:13px">'
        f'Chapter&nbsp;{rec["chapter"]} &nbsp;·&nbsp; '
        f'Pages&nbsp;{wp_str} &nbsp;·&nbsp; '
        f'<code style="font-size:12px">{rec["record_id"]}</code>'
        f'</span>'
        f'</div>'
    )
    st.markdown(header_html, unsafe_allow_html=True)

    # ── Question ──────────────────────────────────────────────────────────────
    st.markdown("#### ❓ Question")
    q_html = (
        f'<div style="background:#161b22;border-left:4px solid #4a9eff;'
        f'padding:12px 16px;border-radius:4px;font-size:15px;'
        f'line-height:1.7;color:#e6edf3">'
        f'{rec["question"]}'
        f'</div>'
    )
    st.markdown(q_html, unsafe_allow_html=True)
    st.markdown("")

    # ── Two-column main layout ────────────────────────────────────────────────
    left, right = st.columns([1, 1], gap="large")

    # ─────────────────────────── LEFT ────────────────────────────────────────
    with left:

        # ── TokenSmith Answer ─────────────────────────────────────────────────
        st.markdown("#### 🤖 TokenSmith Answer")

        corr_with = j.get("answer_correctness", {}).get("with_reference", {})
        corr_no   = j.get("answer_correctness", {}).get("without_reference", {})
        score_w   = corr_with.get("score")
        score_n   = corr_no.get("score")

        if score_w is not None:
            sc  = SCORE_COLOUR.get(score_w, "#666")
            sl  = SCORE_LABEL.get(score_w, str(score_w))
            sn_col = SCORE_COLOUR.get(score_n, "#666")
            sn_lbl = SCORE_LABEL.get(score_n, str(score_n))
            rating_html = (
                f'<div style="display:flex;gap:8px;margin-bottom:8px;flex-wrap:wrap">'
                + badge(f"With ref: {sl}", sc) + " "
                + badge(f"No ref: {sn_lbl}", sn_col)
                + f'</div>'
            )
            st.markdown(rating_html, unsafe_allow_html=True)

        # Strip <<<ANSWER>>> markers
        answer_clean = re.sub(r"<<<[A-Z_]+>>>", "", rec["ts_answer"]).strip()
        answer_html = (
            f'<div style="background:#0d1117;border:1px solid #30363d;'
            f'border-radius:6px;padding:12px 16px;font-size:13px;'
            f'line-height:1.7;color:#e6edf3;white-space:pre-wrap">'
            f'{answer_clean}'
            f'</div>'
        )
        st.markdown(answer_html, unsafe_allow_html=True)

        # Explanations
        exp_w = corr_with.get("explanation", "")
        if exp_w:
            with st.expander("💬 Judge explanation (with reference)", expanded=False):
                st.markdown(exp_w)
                strengths = corr_with.get("strengths", [])
                weaknesses = corr_with.get("weaknesses", [])
                if strengths:
                    st.markdown("**Strengths:**")
                    for s in strengths:
                        st.markdown(f"- {s}")
                if weaknesses:
                    st.markdown("**Weaknesses:**")
                    for w in weaknesses:
                        st.markdown(f"- {w}")

        exp_n = corr_no.get("explanation", "")
        if exp_n:
            with st.expander(f"💬 Judge explanation (no reference) — score: {score_n}", expanded=False):
                st.markdown(exp_n)

        # ── Faithfulness ──────────────────────────────────────────────────────
        st.markdown("#### 🔒 Faithfulness")
        faith    = j.get("faithfulness", {})
        fverdict = faith.get("verdict", "uncertain")
        fc       = VERDICT_COLOUR.get(fverdict, "#666")
        ficon    = VERDICT_ICON.get(fverdict, "❓")
        flabel   = fverdict.replace("_", " ").title()

        st.markdown(badge(f"{ficon} {flabel}", fc), unsafe_allow_html=True)

        fexp = faith.get("explanation", "")
        if fexp:
            with st.expander("💬 Faithfulness explanation", expanded=False):
                st.markdown(fexp)

        unsupported = faith.get("unsupported_claims", [])
        if unsupported:
            with st.expander(f"⚠️ Unsupported claims ({len(unsupported)})", expanded=False):
                for claim in unsupported:
                    st.markdown(f"- {claim}")

        # ── Mock Answer ───────────────────────────────────────────────────────
        with st.expander("📖 Reference mock answer", expanded=False):
            st.markdown(rec["mock_answer"])

    # ─────────────────────────── RIGHT ───────────────────────────────────────
    with right:

        # ── Rubric ────────────────────────────────────────────────────────────
        st.markdown("#### 📋 Rubric")

        rubric_results = (
            j.get("rubric_satisfaction", {}).get("all")
            or j.get("rubric_satisfaction", {}).get("individual")
            or []
        )
        # Map criterion text → judge result
        rubric_judge = {rr.get("criterion", ""): rr for rr in rubric_results}

        for ri, crit in enumerate(rec["rubric"]):
            flags   = rec["rubric_flags"][ri] if ri < len(rec["rubric_flags"]) else {}
            rr      = rubric_judge.get(crit, {})
            verdict = rr.get("verdict", "")
            reason  = rr.get("reason", "")
            rc      = VERDICT_COLOUR.get(verdict, "#30363d")
            icon    = VERDICT_ICON.get(verdict, "❓")

            # Flag badges
            flag_parts = []
            if flags.get("gold"):            flag_parts.append(badge("Gold", "#1a5276"))
            if flags.get("optional"):        flag_parts.append(badge("Optional", "#444"))
            if flags.get("example_analogy"): flag_parts.append(badge("Example", "#6a0572"))
            flag_html = " ".join(flag_parts)

            verdict_html = badge(f"{icon} {verdict.replace('_',' ').title()}", rc) if verdict else ""

            reason_html = ""
            if reason:
                reason_html = (
                    f'<div style="color:#8b949e;font-size:12px;'
                    f'margin-top:5px;font-style:italic">{reason}</div>'
                )

            flags_div = ""
            if flag_html:
                flags_div = f'<div style="margin-top:4px">{flag_html}</div>'

            inner = (
                f'<div style="display:flex;gap:8px;align-items:flex-start">'
                f'<div style="flex:1">'
                f'<div style="font-size:13px;line-height:1.5;color:#e6edf3">{crit}</div>'
                f'{flags_div}'
                f'<div style="margin-top:4px">{verdict_html}</div>'
                f'{reason_html}'
                f'</div>'
                f'</div>'
            )
            st.markdown(card(inner, border_colour=rc), unsafe_allow_html=True)

        # ── Gold Chunks ───────────────────────────────────────────────────────
        st.markdown("#### 🎯 Gold Chunks")

        n_found = sum(1 for gs in rec["gold_status"] if gs["found"])
        n_gold  = len(rec["gold_status"])
        cov_col = "#2d6a4f" if n_found == n_gold else ("#b5770c" if n_found > 0 else "#a4262c")
        cov_html = badge(f"{n_found}/{n_gold} retrieved", cov_col)
        st.markdown(
            f'<div style="margin-bottom:8px">Coverage: {cov_html}</div>',
            unsafe_allow_html=True,
        )

        for ci, gs in enumerate(rec["gold_status"]):
            flags   = rec["gold_chunk_flags"][ci] if ci < len(rec["gold_chunk_flags"]) else {}
            found   = gs["found"]
            bc      = "#2d6a4f" if found else "#a4262c"
            icon    = "🟢" if found else "🔴"
            rank_note = f" &nbsp;<span style='color:#888;font-size:11px'>(rank {gs['rank']})</span>" if gs["rank"] else ""

            flag_parts = []
            if flags.get("gold"):            flag_parts.append(badge("Gold", "#1a5276"))
            if flags.get("optional"):        flag_parts.append(badge("Optional", "#444"))
            if flags.get("example_analogy"): flag_parts.append(badge("Example", "#6a0572"))
            if flags.get("confusing"):       flag_parts.append(badge("Confusing", "#7d3c98"))
            flag_html = " ".join(flag_parts)

            flags_div = f'<div style="margin-top:4px">{flag_html}</div>' if flag_html else ""
            chunk_escaped = gs["chunk"].replace("<", "&lt;").replace(">", "&gt;")

            inner = (
                f'<div style="font-size:12px;font-family:monospace;'
                f'line-height:1.5;color:#e6edf3">'
                f'{icon} {chunk_escaped}{rank_note}'
                f'</div>'
                f'{flags_div}'
            )
            st.markdown(card(inner, border_colour=bc), unsafe_allow_html=True)

    # ── Retrieved Chunks (full width) ─────────────────────────────────────────
    st.markdown("#### 📦 Retrieved Chunks")

    chunk_rel = j.get("chunk_relevance", {})
    rel_items = chunk_rel.get("group") or chunk_rel.get("individual") or []
    rel_by_rank = {rr.get("rank"): rr for rr in rel_items}

    retrieved = rec["retrieved_chunks"]
    n_chunks  = len(retrieved)

    for row_start in range(0, n_chunks, 2):
        row = retrieved[row_start: row_start + 2]
        cols = st.columns(len(row))
        for col, chunk in zip(cols, row):
            rank    = chunk.get("rank", "?")
            rr      = rel_by_rank.get(rank, {})
            verdict = rr.get("verdict", "")
            reason  = rr.get("reason", "")
            rc      = VERDICT_COLOUR.get(verdict, "#30363d")
            icon    = VERDICT_ICON.get(verdict, "")
            v_label = verdict.replace("_", " ").title()

            faiss = chunk.get("faiss_score", 0)
            bm25  = chunk.get("bm25_score", 0)

            content_escaped = (
                chunk.get("content", "")
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )

            # Build each section separately to avoid nested f-strings
            title_row = (
                f'<div style="display:flex;justify-content:space-between;'
                f'align-items:center;margin-bottom:6px">'
                f'<span style="font-weight:700;color:#4a9eff">Rank {rank}</span>'
                f'<span style="font-size:11px;color:#888">'
                f'FAISS: {faiss:.2e} &nbsp; BM25: {bm25:.1f}'
                f'</span>'
                f'</div>'
            )

            verdict_row = ""
            if verdict:
                verdict_row = (
                    f'<div style="margin-bottom:6px">'
                    + badge(f"{icon} {v_label}", rc)
                    + f'</div>'
                )

            reason_row = ""
            if reason:
                reason_row = (
                    f'<div style="color:#8b949e;font-size:11px;'
                    f'margin-top:6px;font-style:italic">{reason}</div>'
                )

            content_div = (
                f'<div style="font-size:12px;font-family:monospace;'
                f'line-height:1.5;color:#e6edf3;'
                f'max-height:180px;overflow-y:auto">'
                f'{content_escaped}'
                f'</div>'
            )

            full_html = (
                f'<div style="background:#0d1117;border:1px solid {rc};'
                f'border-radius:6px;padding:10px 14px">'
                + title_row
                + verdict_row
                + content_div
                + reason_row
                + f'</div>'
            )
            col.markdown(full_html, unsafe_allow_html=True)

    # ── Error banner ──────────────────────────────────────────────────────────
    if rec.get("error"):
        st.error(f"**Error during run:** {rec['error'][:300]}")


# ─────────────────────────────────────────────────────────────────────────────
# Main app
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="TokenSmith Benchmark Viewer",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=DM+Sans:wght@400;500;600&display=swap');
        html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
        code, pre { font-family: 'JetBrains Mono', monospace !important; }
        .block-container { padding-top: 1.5rem; max-width: 1400px; }
        h1 { font-size: 1.5rem !important; font-weight: 600 !important; }
        h4 { font-size: 0.85rem !important; font-weight: 600 !important;
             letter-spacing: 0.05em; text-transform: uppercase;
             color: #8b949e !important; margin-bottom: 8px !important; }
        .stExpander { border: 1px solid #30363d !important; border-radius: 6px !important; }
        hr { border-color: #30363d !important; margin: 1.5rem 0 !important; }
    </style>
    """, unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🔬 Benchmark Viewer")
        st.markdown("---")

        run_dirs = list_run_dirs()
        if not run_dirs:
            st.error(f"No run directories found in:\n`{RESULTS_ROOT}`")
            st.stop()

        run_names = [d.name for d in run_dirs]
        selected_run = st.selectbox("Run", run_names, label_visibility="collapsed")
        run_path = RESULTS_ROOT / selected_run

        records = load_run(str(run_path))
        if not records:
            st.error("No results found in this run directory.")
            st.stop()

        n = len(records)
        n_err = sum(1 for r in records if r.get("error"))
        st.markdown(f"**{n} questions evaluated**")
        if n_err:
            st.warning(f"{n_err} had errors")

        st.markdown("---")
        st.subheader("Filters")

        chapters  = sorted({r["chapter"] for r in records if r["chapter"] is not None})
        ch_opts   = ["All"] + [f"C{c:02d}" for c in chapters]
        ch_filter = st.selectbox("Chapter", ch_opts)

        diff_filter = st.selectbox("Difficulty", ["All", "easy", "medium", "hard"])
        gen_filter  = st.selectbox("Gen status", ["All", "passed", "manual_review"])

        cov_filter = st.selectbox(
            "Gold coverage",
            ["All", "Full (100%)", "Partial", "None (0%)"],
        )
        corr_filter = st.selectbox(
            "Correctness",
            ["All", "✅ Correct (1)", "⚠️ Partial (0)", "❌ Wrong (-1)"],
        )
        faith_filter = st.selectbox(
            "Faithfulness",
            ["All", "Faithful", "Partially faithful", "Unfaithful", "Uncertain"],
        )

        st.markdown("---")
        if "selected_idx" not in st.session_state:
            st.session_state.selected_idx = 0

    # ── Apply filters ─────────────────────────────────────────────────────────
    visible = []
    for i, r in enumerate(records):
        if ch_filter != "All" and r["chapter"] != int(ch_filter.replace("C", "")):
            continue
        if diff_filter != "All" and r["difficulty"] != diff_filter:
            continue
        if gen_filter != "All" and r["generation_status"] != gen_filter:
            continue
        if cov_filter != "All":
            nf = sum(1 for gs in r["gold_status"] if gs["found"])
            ng = len(r["gold_status"])
            if cov_filter == "Full (100%)" and nf < ng:          continue
            if cov_filter == "None (0%)"   and nf > 0:           continue
            if cov_filter == "Partial"     and not (0 < nf < ng): continue
        if corr_filter != "All":
            sc = r["judgements"].get("answer_correctness", {}).get("with_reference", {}).get("score")
            if corr_filter == "✅ Correct (1)" and sc != 1:  continue
            if corr_filter == "⚠️ Partial (0)" and sc != 0:  continue
            if corr_filter == "❌ Wrong (-1)"  and sc != -1: continue
        if faith_filter != "All":
            fv = r["judgements"].get("faithfulness", {}).get("verdict", "")
            mapping = {
                "Faithful": "faithful", "Partially faithful": "partially_faithful",
                "Unfaithful": "unfaithful", "Uncertain": "uncertain",
            }
            if fv != mapping.get(faith_filter, ""):
                continue
        visible.append(i)

    # ── Main area ─────────────────────────────────────────────────────────────
    st.markdown(f"## TokenSmith Benchmark — `{selected_run}`")

    if not visible:
        st.info("No records match the current filters.")
        st.stop()

    st.markdown(
        f'<span style="color:#8b949e;font-size:13px">'
        f'{len(visible)} of {n} questions shown</span>',
        unsafe_allow_html=True,
    )
    st.markdown("")

    # ── Question list ─────────────────────────────────────────────────────────
    header_cols = st.columns([0.5, 0.5, 0.7, 4.2, 0.9, 1.1, 1.0, 1.1])
    for col, lbl in zip(header_cols, ["#", "Ch", "Diff", "Question",
                                       "Coverage", "Correct", "Faith", ""]):
        col.markdown(
            f'<span style="color:#8b949e;font-size:11px;font-weight:700;'
            f'text-transform:uppercase;letter-spacing:0.05em">{lbl}</span>',
            unsafe_allow_html=True,
        )
    st.markdown(
        '<hr style="margin:4px 0;border-color:#30363d">',
        unsafe_allow_html=True,
    )

    sel = st.session_state.selected_idx
    if sel not in visible:
        sel = visible[0]
        st.session_state.selected_idx = sel

    for i in visible:
        r    = records[i]
        diff = r["difficulty"]
        dc   = DIFF_COLOUR.get(diff, "#666")

        nf  = sum(1 for gs in r["gold_status"] if gs["found"])
        ng  = len(r["gold_status"])
        cov_c = "#2d6a4f" if nf == ng else ("#b5770c" if nf > 0 else "#a4262c")

        sc    = r["judgements"].get("answer_correctness", {}).get("with_reference", {}).get("score")
        sc_c  = SCORE_COLOUR.get(sc, "#666")
        sc_s  = {1: "✅ 1", 0: "⚠️ 0", -1: "❌ -1"}.get(sc, "—")

        fv    = r["judgements"].get("faithfulness", {}).get("verdict", "")
        fv_c  = VERDICT_COLOUR.get(fv, "#666")
        fv_s  = {"faithful": "✅", "partially_faithful": "⚠️",
                 "unfaithful": "❌", "uncertain": "❓"}.get(fv, "—")

        is_sel = (sel == i)
        bg = "background:#161b22;" if is_sel else ""

        row = st.columns([0.5, 0.5, 0.7, 4.2, 0.9, 1.1, 1.0, 1.1])

        row[0].markdown(
            f'<div style="{bg}padding:3px;font-size:12px;color:#8b949e">{i}</div>',
            unsafe_allow_html=True)
        row[1].markdown(
            f'<div style="{bg}padding:3px;font-size:12px">C{r["chapter"]:02d}</div>',
            unsafe_allow_html=True)
        row[2].markdown(
            f'<div style="{bg}padding:3px">'
            f'<span style="color:{dc};font-size:12px;font-weight:600">{diff}</span>'
            f'</div>',
            unsafe_allow_html=True)
        truncated_q = r["question"][:80] + ("…" if len(r["question"]) > 80 else "")
        row[3].markdown(
            f'<div style="{bg}padding:3px;font-size:13px">{truncated_q}</div>',
            unsafe_allow_html=True)
        row[4].markdown(
            f'<div style="{bg}padding:3px">'
            f'<span style="color:{cov_c};font-size:13px;font-weight:600">'
            f'{nf}/{ng}</span></div>',
            unsafe_allow_html=True)
        row[5].markdown(
            f'<div style="{bg}padding:3px">'
            f'<span style="color:{sc_c};font-size:13px;font-weight:600">'
            f'{sc_s}</span></div>',
            unsafe_allow_html=True)
        row[6].markdown(
            f'<div style="{bg}padding:3px">'
            f'<span style="color:{fv_c};font-size:13px">{fv_s}</span></div>',
            unsafe_allow_html=True)

        btn_lbl = "✦ Open" if is_sel else "▶ Open"
        if row[7].button(btn_lbl, key=f"view_{i}"):
            st.session_state.selected_idx = i
            st.rerun()

    # ── Detail panel ──────────────────────────────────────────────────────────
    st.markdown("---")
    if sel < len(records):
        render_record(records[sel])


if __name__ == "__main__":
    main()