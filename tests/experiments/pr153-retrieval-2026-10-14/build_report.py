"""Render the paired benchmark's saved results as a self-contained HTML report."""
import html
import json
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
LABELS = {'hybrid': 'Current hybrid', 'vector': 'Current vector-only',
          'pr_as_written': 'PR as written', 'pr_array_repaired': 'PR with array-safe cosine'}


def esc(value):
    return html.escape(str(value))


def table(headers, rows):
    return '<div class="scroll"><table><thead><tr>' + ''.join(f'<th>{esc(h)}</th>' for h in headers) + \
        '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(f'<td>{c}</td>' for c in row) + '</tr>' for row in rows) + \
        '</tbody></table></div>'


def main():
    data = json.loads((HERE / 'results.json').read_text())
    summaries = data['summary']
    rows = data['results']
    main_rows = [s for s in summaries if s['k'] == 8 and s['budget'] == 'equal_k']
    best = main_rows[0]
    baseline = next(s for s in main_rows if s['method'] == 'hybrid')
    pr = next(s for s in main_rows if s['method'] == 'pr_as_written')
    fixed = next(s for s in main_rows if s['method'] == 'pr_array_repaired')
    parts = ['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
        '<title>TokenSmith PR 153 retrieval benchmark</title>',
        '''<style>
        :root{font-family:system-ui,sans-serif;color:#172334;background:#f3f5f8;line-height:1.6}
        body{max-width:1160px;margin:40px auto;padding:0 24px 80px} h1{font-size:34px;line-height:1.2}
        h2{margin-top:38px;font-size:23px} h3{font-size:18px} .eyebrow{font-size:13px;letter-spacing:.09em;text-transform:uppercase;color:#52627b}
        .card,details{background:white;border:1px solid #d9e0e8;border-radius:10px;padding:20px 24px;margin:18px 0}
        .lead{font-size:20px;max-width:950px} .muted{color:#536176} .scroll{overflow-x:auto}
        table{border-collapse:collapse;width:100%;background:#fff;font-size:14px} th,td{text-align:left;border-bottom:1px solid #e0e5ed;padding:12px 14px;vertical-align:top}
        th{background:#eaf0f7;font-weight:650} .pass{color:#13633e;font-weight:650}.fail{color:#a02c32;font-weight:650}
        summary{cursor:pointer;font-weight:650} a{color:#1558a1} pre{white-space:pre-wrap;overflow-wrap:anywhere;font:13px/1.6 ui-monospace,monospace}
        li{margin:7px 0} code{font-size:.9em} .evidence{border-left:3px solid #a8b9d1;padding-left:14px;white-space:pre-wrap;font-size:13px;max-height:420px;overflow:auto}
        </style>''', '<body><p class="eyebrow">TokenSmith · measured 14 October 2026</p>',
        '<h1>Does graph search improve retrieval?</h1>',
        f'<p class="lead">At eight returned sources, current hybrid search passes <b>{baseline["passed"]}/16</b> evidence checks. '
        f'The PR passes <b>{pr["passed"]}/16</b>; fixing its array-similarity bug yields <b>{fixed["passed"]}/16</b>.</p>',
        '<p class="muted">1,574 BuzzDB textbook chunks · 10 conceptual and 6 multi-evidence questions · identical Nomic passage embeddings · three timing passes</p>']
    parts.append('<h2>Equal source budget</h2><p>Every method is evaluated on its first eight sources. '
        'The PR natively returns sixteen for a requested limit of eight; its full output is evaluated separately below.</p>')
    parts.append(table(['Method', 'Passed / 16', 'Conceptual / 10', 'Multi-evidence / 6', 'Median search', 'Mean context'],
        [[esc(LABELS[s['method']]), f"{s['passed']}/16", str(s['conceptual_passed']), str(s['multi_evidence_passed']),
          f"{s['search_seconds']['median']*1000:,.0f} ms", f"{s['mean_context_chars']:,.0f} characters"] for s in main_rows]))
    parts.append('<p class="muted">Search timings exclude embedding service time. They include the PR’s JSON graph reads, traversal, and HTML graph generation. '
                 'The PR still computes/selects its native sixteen results before we score its first eight, so timing is for its actual implementation. '
                 'Embedding service and the baseline index were warmed; the first graph call may include lazy plotting setup. These are local search timings, not end-to-end answer latency.</p>')
    parts.append('<h2>Budget sensitivity</h2>')
    budget_labels = {'native': 'Native output', 'equal_k': 'First K sources', 'equal_k_and_chars': 'First K, capped at hybrid context characters'}
    parts.append(table(['Requested K', 'Method', 'Scoring budget', 'Passed / 16', 'Mean sources', 'Mean characters'],
        [[str(s['k']), esc(LABELS[s['method']]), esc(budget_labels[s['budget']]), f"{s['passed']}/16",
          f"{s['mean_source_count']:.1f}", f"{s['mean_context_chars']:,.0f}"]
         for s in summaries if s['method'] != 'vector' or s['budget'] == 'equal_k']))
    parts.append('<p class="muted">Character-capped results use each question’s hybrid context size as the budget, '
        'preserve retrieval order, and truncate the final passage if needed. This is a sensitivity check, not the app’s token packer.</p>')
    parts.append('<h2>Per-question results at eight sources</h2>')
    case_ids = [r['id'] for r in rows if r['method'] == 'hybrid' and r['k'] == 8]
    lookup = {(r['method'], r['k'], r['id']): r for r in rows}
    methods = list(LABELS)
    matrix = []
    for case_id in case_ids:
        line = [f'<a href="#{esc(case_id)}">{esc(case_id)}</a>']
        for method in methods:
            score = lookup[(method, 8, case_id)]['equal_k']
            line.append('<span class="pass">PASS</span>' if score['passed'] else '<span class="fail">FAIL</span>')
        matrix.append(line)
    parts.append(table(['Question'] + list(LABELS.values()), matrix))
    parts.append('<h2>What the graph actually contains</h2>')
    parts.append(table(['Graph', 'Directed edges', 'Possible edges used', 'Average neighbors', 'Build time', 'JSON size'],
        [[esc(LABELS[name]), f"{g['directed_edges']:,}", f"{g['density']*100:.2f}%", f"{g['out_degree']['mean']:,.1f}",
          f"{g['seconds']:.2f} s", f"{g['disk_bytes']/1e6:.1f} MB"] for name,g in data['graph_builds'].items()]))
    parts.append('<p>The submitted similarity helper rejects NumPy arrays, and graph construction swallows that error. '
        'Its graph therefore uses term-overlap links for stored embeddings. The diagnostic fix makes the same cosine formula array-safe. '
        'With the actual 0.1 cosine threshold, the corrected graph is complete: every chunk connects to every other chunk. '
        'Thresholds, traversal depth, ranking weights, query suffix, and output limits were not tuned to these questions.</p>')
    parts.append('<h2>Method and limitations</h2><ul>'
        '<li>Current baseline: the working-tree production <code>search_library</code>, including hybrid rank fusion, source expansion, and source selection. '
        'The vector control uses its existing <code>searchMode=vector</code>.</li>'
        '<li>PR baseline: exact engine and storage source at <code>10df6433a06bde4fcbaa2977fb1767c82e8139d2</code>. '
        'A freshly built benchmark graph avoids the separately verified existing-index migration failure.</li>'
        '<li>Diagnostic variant: only <code>embedding_similarity</code> is replaced with an array-safe float64 NumPy implementation. '
        'It agrees with the original list-based formula to floating-point precision on 100 sampled pairs. '
        'Its indexing time uses the faster numerical reductions and should not be read as the time of a one-line truth-value fix.</li>'
        '<li>Identical isolated SQLite/FAISS fixture, 1,574 chunks, 768-dimensional <code>nomic-embed-text:latest</code>, '
        'pinned model digest and Ollama 0.31.1. The existing passage cache was validated against the corpus, parsed chunks, and embedding function.</li>'
        '<li>Real query embeddings were computed once per exact query string and reused to remove model-service timing noise. '
        'The PR’s literal query transformation, including the missing space before “in detail”, is preserved. '
        'Current and PR methods share passage vectors but intentionally use their own actual query text.</li>'
        '<li>Three interleaved timing passes at K=8 and K=4. Retrieved IDs were asserted identical across repeats. '
        'No embedding-provider fallback was permitted; vector hits were checked explicitly.</li>'
        '<li>Scoring uses the repository’s authored expected chunks, required evidence groups, required phrases, and forbidden content. '
        'This suite is small, from one textbook, and already used for hybrid regression testing; it is not an independent holdout. '
        'It does not measure generated-answer accuracy. Full source text is saved for review.</li>'
        '<li>No app code, app settings, student documents, or PR files were changed. '
        'Dependencies were installed only into <code>tmp/pr153-benchmark-deps</code>.</li></ul>')
    parts.append('<h2>Concrete failures and retrieved evidence</h2>')
    for case_id in case_ids:
        base = lookup[('hybrid', 8, case_id)]
        parts.append(f'<details id="{esc(case_id)}"><summary>{esc(case_id)}</summary><p>{esc(base["question"])}</p>')
        for method in methods:
            row = lookup[(method, 8, case_id)]
            score = row['equal_k']
            parts.append(f'<h3>{esc(LABELS[method])}: {"PASS" if score["passed"] else "FAIL"}</h3>')
            if row['error']:
                parts.append(f'<pre>{esc(row["error"])}</pre>')
            if score['failures']:
                parts.append('<ul>' + ''.join(f'<li>{esc(f)}</li>' for f in score['failures']) + '</ul>')
            parts.append(f'<p class="muted">{score["source_count"]} sources · {score["context_chars"]:,} characters</p>')
            for rank, source in enumerate(score['evidence'], 1):
                parts.append(f'<details><summary>{rank}. {esc(", ".join(source["chunkIds"]))} — {esc(source["section"] or "")}</summary>'
                             f'<div class="evidence">{esc(source["context"])}</div></details>')
        parts.append('</details>')
    parts.append('<h2>Reproduce and audit</h2><p><a href="results.json">Raw results and provenance</a> · '
        '<a href="query-embeddings.json">Query vectors</a> · <a href="environment.json">Dependency versions</a> · '
        '<a href="README.md">Reproduction instructions</a> · <a href="run.py">Portable replay runner</a> · '
        '<a href="run_original.py">Original runner</a> · <a href="source/pr/tokensmith_engine.py">PR engine snapshot</a></p>'
        '<pre>python3.12 -m venv .venv-pr153\n'
        '.venv-pr153/bin/python -m pip install -r tests/experiments/pr153-retrieval-2026-10-14/requirements.txt\n'
        '.venv-pr153/bin/python tests/experiments/pr153-retrieval-2026-10-14/run.py</pre>')
    parts.append('<details><summary>Recorded metadata and preflight checks</summary><pre>' + esc(json.dumps(
        {'metadata': data['metadata'], 'preflight': data['preflight']}, indent=2)) + '</pre></details></body></html>')
    (HERE / 'report.html').write_text('\n'.join(parts))
    print(HERE / 'report.html')


if __name__ == '__main__':
    main()
