"""Render captured results plus the separate, evidence-based qualitative review."""
import json
import hashlib
from pathlib import Path
from html import escape as e

base = Path(__file__).resolve().parent
data = json.loads((base / 'results.json').read_text())
review = json.loads((base / 'review.json').read_text())
assert hashlib.sha256((base / 'results.json').read_bytes()).hexdigest() == review['reviewedResultsSha256'], 'These results need a new semantic review before rendering.'
verification = json.loads((base / 'verification.json').read_text())
judgments = {(r['topic'], r['number']): r for r in review['questions']}
book = Path(verification['bookPath']).read_text().splitlines()
rows = [q for t in data['topics'] for q in t['questions']]
good = [r for r in rows if 'question' in r]
citations = [c for q in verification['questions'] for c in q['citations']]
assert len(judgments) == len(rows) == 12
assert all(c['foundInOriginalBookLines'] for c in citations)
assert all(q['gradesConsistentWithSavedKey'] for q in verification['questions'])

parts = ['''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>BuzzDBBook · True/False question audit</title><style>
:root{color-scheme:light}*{box-sizing:border-box}body{font:17px/1.6 system-ui,sans-serif;color:#21332e;background:#f7f7f1;margin:0}main{max-width:1040px;padding:48px 28px 80px;margin:auto}h1{font-size:36px;line-height:1.2;margin:0 0 16px}h2{font-size:25px;margin:42px 0 18px}h3{font-size:20px;margin:0 0 14px}p{margin:12px 0}.meta{color:#55655e}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:28px 0}.stat,article{border:1px solid #cbd5cb;background:#fffef9;border-radius:12px;padding:22px}.stat strong{display:block;font-size:29px}article{margin:18px 0}.tag{font-size:14px;font-weight:650;padding:3px 10px;border-radius:20px;display:inline-block;background:#dcecdf}.Qualify{background:#fff0c3;color:#644a0e}.Blocked{background:#f6dfdb;color:#752d22}.question{font-size:20px;font-weight:550}.key{font-weight:700}details{border-top:1px solid #d8ded5;padding:12px 0 0;margin-top:18px}summary{cursor:pointer;color:#355947}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.6 ui-monospace,monospace;background:#f3f4ed;padding:16px;border-radius:8px}blockquote{border-left:3px solid #849b87;padding:0 16px;margin:16px 0}.rewrite{background:#f4f1df;padding:14px;border-radius:6px}a{color:#225b45}table{border-collapse:collapse;width:100%;font-size:16px}td,th{text-align:left;border-bottom:1px solid #d1d9cd;padding:10px 12px}footer{margin-top:40px;font-size:14px;color:#58665e}@media(max-width:650px){main{padding:24px 16px}.stats{grid-template-columns:repeat(2,1fr)}h1{font-size:29px}}
</style><main><p class="meta">TokenSmith · 14 October 2026</p><h1>BuzzDBBook: True/False question audit</h1>
<p>Two consecutive questions on each of six topics, using the current app’s retrieval and Practice generation code.</p>
<div class="stats"><div class="stat"><strong>12</strong>questions requested</div><div class="stat"><strong>8</strong>sound as written</div><div class="stat"><strong>3</strong>need qualification</div><div class="stat"><strong>1</strong>blocked for bad citations</div></div>''']
parts.append(f'<p>{e(review["summary"])}</p>')
parts.append('<h2>What this run tested</h2><p>Gemma 4 E4B + Nomic embeddings; hybrid retrieval from an isolated snapshot of the loaded BuzzDBBook library. Same topic preparation, four-passage question selection, parser, one-retry policy, and True/False grading functions as the app. No production prompts or settings were changed during the test, and no test questions were added to your history.</p>')
parts.append('<p>Reasoning was off. Actual requests used an 8,192-token context and a 1,024-token output limit; Practice uses temperature 0.3. Accepted questions took 9.7–20.6 seconds (median 14.2), excluding one-time topic retrieval. The failed Volcano request took 29.5 seconds across two attempts. Eleven accepted questions contained seven True keys and four False keys.</p>')
parts.append('<p>There were 13 full generation attempts and 13 one-token context-sizing probes. All 22 accepted citations were checked against the original book at their stated line locations, allowing presentation-only differences such as Markdown emphasis. Both answer-button outcomes agreed with each saved key. These checks establish provenance and grading consistency; the semantic review below checks whether each key is justified.</p>')
parts.append(f'<p class="meta">{e(review["reviewer"])} {e(review["limitations"])}</p>')
parts.append('<h2>Results by topic</h2><table><tr><th>Topic</th><th>Question 1</th><th>Question 2</th></tr>')
for t in data['topics']:
    parts.append(f'<tr><td>{e(t["title"])}</td>' + ''.join(f'<td>{e(judgments[(t["title"], n)]["verdict"])}</td>' for n in (1,2)) + '</tr>')
parts.append('</table><h2>Exact questions and evidence</h2>')
for t in data['topics']:
    for row in t['questions']:
        r = judgments[(t['title'], row['number'])]
        parts.append(f'<article><h3>{e(t["title"])} · Question {row["number"]} <span class="tag {e(r["verdict"])}">{e(r["verdict"])}</span></h3>')
        if 'question' in row:
            q = row['question']
            parts.append(f'<p class="question">{e(q["text"])}</p><p class="key">Generated key: {q["correctAnswer"]} · {row["seconds"]:.1f}s · {row["calls"]} attempt</p><p>{e(r["note"])}</p>')
            if 'rewrite' in r: parts.append(f'<p class="rewrite"><strong>Suggested wording — not a generated retest:</strong> {e(r["rewrite"])} <strong>Key: {r["rewriteKey"]}</strong></p>')
            parts.append(f'<details><summary>Generated explanation and exact supporting quotations</summary><p>{e(q["referenceAnswer"])}</p>')
            for c in q['rubric']['criteria']:
                parts.append(f'<p><strong>{e(c["id"])}</strong>: {e(c["description"])}</p>')
                for v in c['evidence']: parts.append(f'<blockquote>{e(v["quote"])}</blockquote><p class="meta">{e(v["sourceKey"])}</p>')
            parts.append('</details><details><summary>Full cited passages</summary>')
            for source in q['sources']:
                parts.append(f'<h3>{e(source.get("sectionHeader", ""))}</h3><p class="meta">{e(source.get("chunkId", ""))} · Book lines {source["lineFrom"]}–{source["lineTo"]}</p><pre>{e(source.get("context") or source["excerpt"])}</pre>')
            parts.append('</details>')
        else:
            parts.append(f'<p>{e(r["note"])}</p><p class="meta">{row["seconds"]:.1f}s · {row["calls"]} attempts · Neither candidate was shown as an answerable question.</p><details><summary>Rejected model responses</summary>')
            for c in data['calls']:
                if c['topic'] == t['title'] and c['questionNumber'] == row['number']:
                    parts.append(f'<p>{e(c.get("validation", c.get("error", "")))}</p><pre>{e(c.get("reply", {}).get("text", ""))}</pre>')
            parts.append('</details>')
        parts.append('<details><summary>Additional book context used in this review</summary>')
        for line in r['bookLines']:
            parts.append(f'<p class="meta">Original book, line {line}</p><pre>{e(chr(10).join(book[line-1:line+7]))}</pre>')
        parts.append('</details></article>')
parts.append('<h2>What remains to improve</h2><p>Preserve the cost metric, implementation, and exceptions in the visible statement. Exact quotations alone cannot establish that an unrestricted claim is fair. Also avoid repeating the same learning objective: the heap-file pair covered essentially the same fact twice.</p><p>The failed Volcano case is a citation-assignment error. A possible next improvement is to reassign a quote only when it has one unambiguous exact match among the passages supplied to the model, then revalidate it. This could avoid a model retry for that narrow failure; it would not solve semantic overstatement.</p>')
parts.append(f'<footer><p>Files: <a href="results.json">raw generations and runtime settings</a> · <a href="review.json">qualitative review</a> · <a href="verification.json">original-book citation checks</a> · <a href="run.mjs">reproducible runner</a></p><p>Book SHA-256: {e(verification["bookSha256"])}</p></footer></main></html>')
(base / 'report.html').write_text(''.join(parts))
print(base / 'report.html')
