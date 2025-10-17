#!/usr/bin/env python3
"""Render stored retrieval diagnostics into tabular text/HTML."""

import html
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

LOG_PATH = Path("logs/retrieval_diagnostics.json")
TOP_K = 5
SHOW_TEXT = True
OUTPUT_HTML = Path("logs/retrieval_report.html")


def sanitize(text: str) -> str:
    return text.replace("\t", " ").replace("\n", " ").strip()


def format_tsv(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    header_line = "\t".join(headers)
    row_lines = ["\t".join(row) for row in rows]
    return "\n".join([header_line, *row_lines])


def load_payload(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Diagnostics file not found: {path}")
    return json.loads(path.read_text())


def map_records(records: List[Dict[str, object]]) -> Dict[int, Dict[str, object]]:
    mapped: Dict[int, Dict[str, object]] = {}
    for record in records:
        idx = int(record.get("chunk_idx"))
        mapped[idx] = record
    return mapped
 

def build_faiss_rows(
    faiss_records: List[Dict[str, object]],
    bm25_lookup: Dict[int, Dict[str, object]],
) -> Tuple[Sequence[str], List[List[str]]]:
    headers = (
        "FAISS Rank",
        "Chunk",
        "FAISS Score",
        "FAISS Distance",
        "BM25 Rank",
        "BM25 Score",
        "Text",
    )
    rows: List[List[str]] = []
    for record in faiss_records[:TOP_K]:
        idx = int(record.get("chunk_idx"))
        bm25_match = bm25_lookup.get(idx, {})
        bm25_score = bm25_match.get("score") if bm25_match else None
        bm25_rank = bm25_match.get("rank") if bm25_match else None
        rows.append([
            f"#{record.get('rank')}",
            str(idx),
            f"{record.get('score', 0.0):.4f}",
            f"{record.get('distance', 0.0):.4f}",
            f"#{bm25_rank}" if bm25_rank is not None else "-",
            f"{bm25_score:.4f}" if isinstance(bm25_score, (float, int)) else "-",
            sanitize(record.get("text", "") if SHOW_TEXT else ""),
        ])
    return headers, rows


def build_bm25_rows(
    bm25_records: List[Dict[str, object]],
    faiss_lookup: Dict[int, Dict[str, object]],
) -> Tuple[Sequence[str], List[List[str]]]:
    headers = (
        "BM25 Rank",
        "Chunk",
        "BM25 Score",
        "FAISS Rank",
        "FAISS Score",
        "FAISS Distance",
        "Text",
    )
    rows: List[List[str]] = []
    for record in bm25_records[:TOP_K]:
        idx = int(record.get("chunk_idx"))
        faiss_match = faiss_lookup.get(idx, {})
        faiss_rank = faiss_match.get("rank") if faiss_match else None
        rows.append([
            f"#{record.get('rank')}",
            str(idx),
            f"{record.get('score', 0.0):.4f}",
            f"#{faiss_rank}" if faiss_rank is not None else "-",
            f"{faiss_match.get('score', 0.0):.4f}" if faiss_match else "-",
            f"{faiss_match.get('distance', 0.0):.4f}" if faiss_match else "-",
            sanitize(record.get("text", "") if SHOW_TEXT else ""),
        ])
    return headers, rows


def build_html_table(title: str, headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    head_cells = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body_rows = []
    for row in rows:
        cells = "".join(f"<td>{html.escape(cell)}</td>" for cell in row)
        body_rows.append(f"<tr>{cells}</tr>")
    body = "".join(body_rows)
    return (
        f"<h2>{html.escape(title)}</h2>"
        f"<table border=1 cellpadding=4 cellspacing=0>"
        f"<thead><tr>{head_cells}</tr></thead>"
        f"<tbody>{body}</tbody>"
        "</table>"
    )


def main() -> None:
    payload = load_payload(LOG_PATH)
    faiss_records: List[Dict[str, object]] = payload.get("faiss", [])
    bm25_records: List[Dict[str, object]] = payload.get("bm25", [])

    faiss_lookup = map_records(faiss_records)
    bm25_lookup = map_records(bm25_records)

    faiss_headers, faiss_rows = build_faiss_rows(faiss_records, bm25_lookup)
    bm25_headers, bm25_rows = build_bm25_rows(bm25_records, faiss_lookup)

    print(f"Query: {payload.get('query', 'N/A')}")
    print(f"Pool size: {payload.get('pool_size', 'N/A')} | Top-K: {TOP_K}")
    print()
    print("FAISS Top Results (TSV, paste -> Split text to columns)")
    print(format_tsv(faiss_headers, faiss_rows))
    print()
    print("BM25 Top Results (TSV, paste -> Split text to columns)")
    print(format_tsv(bm25_headers, bm25_rows))

    html_report = "\n".join(
        [
            "<html><head><meta charset='utf-8'><title>Retrieval Report</title></head><body>",
            f"<p><b>Query:</b> {html.escape(str(payload.get('query', 'N/A')))}<br>"
            f"<b>Pool size:</b> {payload.get('pool_size', 'N/A')} | <b>Top-K:</b> {TOP_K}</p>",
            build_html_table("FAISS Top Results", faiss_headers, faiss_rows),
            build_html_table("BM25 Top Results", bm25_headers, bm25_rows),
            "</body></html>",
        ]
    )
    OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_HTML.write_text(html_report, encoding="utf-8")
    print()
    print(f"Saved HTML report to {OUTPUT_HTML}")


if __name__ == "__main__":
    main()
