#!/usr/bin/env python3
"""
log_report_generator.py
Generate a human-readable HTML report for a single-query log session.

Usage:
  python log_report_generator.py --session_id 20250918_112547 --out report.html
"""

import argparse
import json
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, select_autoescape
import pandas as pd


def load_session_logs(session_id: str, logs_dir: str = "logs"):
    path = Path(logs_dir) / f"run_{session_id}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Log file not found: {path}")
    logs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                logs.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    return logs


def extract_single_query(logs):
    """Return the first query log (assumes only one)."""
    for entry in logs:
        if entry.get("event") == "query":
            return entry
    raise ValueError("No query found in logs.")


def build_chunk_table(query_log):
    # retrieval candidates
    cand_idxs = query_log.get("retrieval", {}).get("candidate_indices", [])
    faiss_dists = query_log.get("retrieval", {}).get("faiss_distances", {})

    # ranker scores
    ranking = query_log.get("ranking", {})
    ensemble = query_log.get("ensemble", {})
    final_order = ensemble.get("final_ranking", [])
    ranker_names = list(ranking.keys())
    ranking['ensemble'] = {
        "ranks": {}
    }
    for rank, chunk_idx in enumerate(final_order):
        ranking['ensemble']['ranks'][chunk_idx] = rank
    ranker_names.append('ensemble')

    rows = []
    for idx in cand_idxs:
        row = {
            "chunk_idx": idx,
            "faiss_dist": faiss_dists.get(str(idx), faiss_dists.get(idx, None)),
        }
        for r in ranker_names:
            scores = ranking[r].get("scores", {})
            ranks = ranking[r].get("ranks", {})
            row[f"{r}_score"] = scores.get(str(idx), scores.get(idx, None))
            row[f"{r}_rank"] = ranks.get(str(idx), ranks.get(idx, None))
        rows.append(row)

    chunks_used = query_log.get("chunks_used", [])
    idx_to_text = {c["global_index"]: c.get("chunks", "") for c in chunks_used}

    df = pd.DataFrame(rows).sort_values("faiss_dist", ascending=True)
    df["chunk_text"] = df["chunk_idx"].map(idx_to_text.get)
    return df, ranker_names


def render_report(session_id, query_log, chunk_df, ranker_names, out_path):
    env = Environment(
        loader=FileSystemLoader("."), autoescape=select_autoescape(["html", "xml"])
    )

    template_str = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="utf-8"/>
      <title>RAG Report – {{ session_id }}</title>
      <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
      <style>
        body { background: #f8f9fa; }
        table { font-size: 14px; }
        th { background: #e9ecef; }
      </style>
    </head>
    <body>
      <div class="container my-4">
        <h1>📘 RAG Report</h1>
        <h5 class="text-muted">Session: {{ session_id }}</h5>

        <h2>Query</h2>
        <p><b>{{ query_log.query }}</b></p>
          
        <h2>Final LLM Response</h2>
            <div class="card bg-light mb-3">
            <div class="card-body">
                <pre style="white-space: pre-wrap; word-wrap: break-word;">{{ query_log.generation.response_full }}</pre>
            </div>
        </div>
        
        <h2>Query Planner Config Diff</h2>
        <table class="table table-bordered table-sm">
          <thead><tr><th>Parameter</th><th>Old</th><th>New</th></tr></thead>
          <tbody>
          {% for key, vals in query_log.planner.config_diff.items() %}
            <tr>
              <td>{{ key }}</td>
              <td>{{ vals.old }}</td>
              <td>{{ vals.new }}</td>
            </tr>
          {% endfor %}
          </tbody>
        </table>


        <h2>Retrieval</h2>
        <p>Pool size requested: {{ query_log.retrieval.pool_size_requested }}<br/>
           Candidates returned: {{ query_log.retrieval.candidates_returned }}<br/>
           FAISS distances: min={{ query_log.retrieval.faiss_stats.min_distance }},
           avg={{ query_log.retrieval.faiss_stats.avg_distance }},
           max={{ query_log.retrieval.faiss_stats.max_distance }}</p>

        <h2>Ensemble Fusion</h2>
        <p>Method: {{ query_log.ensemble.method }}<br/>
           Weights: {{ query_log.ensemble.weights }}<br/>
           Final ranking (top 10): {{ query_log.ensemble.final_ranking[:10] }}</p>

        <h2>Chunks Table</h2>
        <div class="table-responsive">
        <table class="table table-bordered table-sm">
          <thead>
            <tr>
              <th>Chunk Idx</th>
              <th>FAISS Dist</th>
              {% for r in ranker_names %}
                <th>{{r}} Score</th>
                <th>{{r}} Rank</th>
              {% endfor %}
            </tr>
          </thead>
          <tbody>
          {% for row in chunk_df.to_dict(orient="records") %}
            <tr>
              <td>{{ row.chunk_idx }}</td>
              <td>{{ "%.4f"|format(row.faiss_dist) if row.faiss_dist is not none else "-" }}</td>
              {% for r in ranker_names %}
                <td>{{ "%.4f"|format(row[r + "_score"]) if row[r + "_score"] is not none else "-" }}</td>
                <td>{{ row[r + "_rank"] if row[r + "_rank"] is not none else "-" }}</td>
              {% endfor %}
            </tr>
          {% endfor %}
          </tbody>
        </table>
        </div>

        <h2>Final Chunks Used</h2>
        <ol>
        {% for c in query_log.chunks_used %}
          <li>
            <b>Source:</b> {{ c.source }} (idx={{ c.global_index }})<br/>
            <b>Chunks:</b> {{ c.chunks }}
          </li>
        {% endfor %}
        </ol>
        
        <h2>Discarded Chunks</h2>
        <ol>
        {% for c in query_log.chunks_discarded %}
          <li>
            <b>Source:</b> {{ c.source }} (idx={{ c.global_index }})<br/>
            <b>Chunks:</b> {{ c.chunks }}
          </li>
        {% endfor %}
        </ol>
      </div>
    </body>
    </html>
    """

    template = env.from_string(template_str)

    html = template.render(
        session_id=session_id,
        query_log=query_log,
        chunk_df=chunk_df,
        ranker_names=ranker_names,
        config=query_log.get("planner", {}).get("config_diff", {}),
        base_cfg=query_log.get("planner", {}).get("base_cfg", {}),
        new_cfg=query_log.get("planner", {}).get("new_cfg", {}),
    )

    Path(out_path).write_text(html, encoding="utf-8")
    print(f"✅ Report written to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session_id", required=True)
    parser.add_argument("--out", default="report.html")
    args = parser.parse_args()

    logs = load_session_logs(args.session_id)
    query_log = extract_single_query(logs)
    chunk_df, ranker_names = build_chunk_table(query_log)
    render_report(args.session_id,
                  query_log,
                  chunk_df,
                  ranker_names,
                  args.out)


if __name__ == "__main__":
    main()
