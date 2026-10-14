"""Summarize paired results with concept-cluster intervals and an auditable HTML report."""
import html
import json
import math
import statistics
from pathlib import Path
import numpy as np

OUT=Path(__file__).resolve().parent
read=lambda name:json.loads((OUT/name).read_text())
def write(name,value): (OUT/name).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
def interval(values):
    a=np.asarray(values,dtype=float)
    draws=np.random.default_rng(157).choice(a,(20000,len(a)),replace=True).mean(axis=1)
    return [float(x) for x in np.quantile(draws,[.025,.975])]
def dist(values):
    return {'n':len(values),'median':statistics.median(values),'p95':float(np.percentile(values,95))}
cases=read('cases.json');retrieval=read('retrieval.json');prep=read('preparation.json')
summaries=read('retrieval-summary.json')
for s in summaries:
    subset=[r for r in retrieval if r['mode']==s['mode'] and r['k']==s['k'] and
            (s['kind']=='all' or r['kind'].startswith(s['kind']))]
    deltas=[]
    for concept in sorted({r['concept'] for r in subset}):
        data=[r for r in subset if r['concept']==concept]
        scores={arm:statistics.mean(r[s['metric']] for r in data if r['arm']==arm) for arm in ['off','on']}
        deltas.append(scores['on']-scores['off'])
    s['delta']=statistics.mean(deltas);s['conceptBootstrap95']=interval(deltas)
write('retrieval-summary.json',summaries)

primary=[r for r in retrieval if r['mode']=='hybrid' and r['k']==4]
chapter_results=[]
for chapter in [2,8,11,15]:
    for kind in ['casual','technical']:
        rs=[r for r in primary if r['chapter']==chapter and r['kind'].startswith(kind)]
        chapter_results.append({'chapter':chapter,'kind':kind,'n':len(rs)//2,
            **{arm:sum(r['completeEvidence'] for r in rs if r['arm']==arm) for arm in ['off','on']}})
write('chapter-results.json',chapter_results)
latency=read('search-timings.json')
search_times={arm:dist([r['seconds'] for r in latency if r['arm']==arm]) for arm in ['off','on']}
cost={arm:{run:next(p for p in prep if p['arm']==arm and p['run']==run)['totalSeconds'] for run in ['cold','cached']} for arm in ['off','on']}
extra=cost['on']['cold']-cost['off']['cold']
analysis={'preparationSeconds':cost,'extraInitialSeconds':extra,'warmSearchSeconds':search_times,
          'queryEmbeddingSeconds':dist(read('query-embedding-times.json')),
          'amortizedExtraSecondsPerQuery':{str(n):extra/n for n in [10,100,1000]}}

answer_summary=None
scores_path=OUT/'manual-scores.json'
if scores_path.exists():
    scores=read('manual-scores.json');key=read('blind-key.json');answers=read('answers.json')
    for s in scores:
        s['arm']=key[s['reviewId']];s['id']=s['reviewId'].rsplit('.',1)[0]
    assert len(scores)==48 and len({s['reviewId'] for s in scores})==48
    pairs=[]
    for c in [c for c in cases if c['generateAnswer']]:
        pair={arm:next(s for s in scores if s['id']==c['id'] and s['arm']==arm) for arm in ['off','on']}
        pairs.append({'id':c['id'],'off':pair['off'],'on':pair['on']})
    answer_summary={'n':len(pairs),'rubric':'0 wrong/refusal/unsupported core, 1 partial, 2 both criteria correct',
        'contentCorrect':{arm:sum(p[arm]['score']==2 for p in pairs) for arm in ['off','on']},
        'supportedCorrect':{arm:sum(p[arm]['score']==2 and p[arm]['supported'] for p in pairs) for arm in ['off','on']},
        'supportedPartialOrComplete':{arm:sum(p[arm]['score']>=1 and p[arm]['supported'] for p in pairs) for arm in ['off','on']},
        'scoreHistogram':{arm:{str(score):sum(p[arm]['score']==score for p in pairs) for score in [0,1,2]} for arm in ['off','on']},
        'meanScore':{arm:statistics.mean(p[arm]['score'] for p in pairs) for arm in ['off','on']},
        'times':{arm:dist([a['seconds'] for a in answers if a['arm']==arm]) for arm in ['off','on']},
        'errors':{arm:sum(bool(a.get('error')) for a in answers if a['arm']==arm) for arm in ['off','on']}}
    d=[int(p['on']['score']==2 and p['on']['supported'])-int(p['off']['score']==2 and p['off']['supported']) for p in pairs]
    answer_summary.update(wins=d.count(1),losses=d.count(-1),delta=statistics.mean(d),bootstrap95=interval(d))
    gains=answer_summary['delta']
    analysis['breakEvenQueriesAtHypotheticalMinutesSaved']={str(m):(extra/(gains*m*60) if gains>0 else None) for m in [1,3,5]}
    write('scored-answer-pairs.json',pairs);write('answer-summary.json',answer_summary)
analysis['answerSummary']=answer_summary
write('analysis.json',analysis)

esc=lambda s:html.escape(str(s))
def table(headers,rows):
    return '<table><thead><tr>'+''.join('<th>'+esc(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(c)+'</td>' for c in row)+'</tr>' for row in rows)+'</tbody></table>'
body=['<h1>Does plain-language enrichment earn its preparation cost?</h1>',
      '<p class="subtitle">Independent local evaluation of TokenSmith PR #157 · 14 October 2026</p>',
      '<p><strong>Recommendation: keep it optional and off by default.</strong> The observed answer gain is small and uncertain. The extra preparation may be reasonable for a course library used repeatedly, especially when it can run in the background; this result does not justify making a student wait for it on every new document.</p>',
      '<p>72 newly authored student-style questions, 24 concepts, four complete textbook chapters. Questions and gold passages were frozen before description generation. Every eligible passage was enriched, including distractors. This is a synthetic holdout within one textbook; it is not a study of real student learning.</p>',
      '<p><a href="https://github.com/georgia-tech-db/TokenSmith/pull/157">PR #157</a> at <code>d628f8322c09911306c923d655db1139aab035e0</code>. Installed Gemma 4 E4B and Nomic on Apple M1 Max, 32 GB. Both arms use the exact same PR engine with the option off/on.</p>',
      '<h2>Preparation cost</h2>',
      table(['Run','Off','On','Extra'],[[run,f'{cost["off"][run]:.1f} s',f'{cost["on"][run]:.1f} s',f'{cost["on"][run]-cost["off"][run]:.1f} s'] for run in ['cold','cached']]),
      '<p>There were 416 original passages and 251 eligible passages. All 251 descriptions were accepted. Description generation took 717.5 seconds (median 2.80 seconds per call); embedding took 16.5 seconds off and 20.1 seconds on. Unchanged cached preparations made no model calls.</p>',
      '<p>“Cold” means empty application caches, not cleared OS caches. Off ran before on. Repeat runs verify reuse of cached descriptions and embeddings. Model startup time is included. Both indexes contain identical original source text and passage counts.</p>',
      '<p>Ollama reported cumulative model-load durations of 7.8 seconds off and 196.0 seconds on across all requests. These are included in the totals, not an extra charge. Within the description calls, reported prompt evaluation and generation took 185.3 and 342.5 seconds respectively. This is one measured fresh preparation per arm on this machine, not a repeated hardware performance benchmark.</p>',
      '<h2>Retrieval: production hybrid, four sources</h2>',
      table(['Question set','Metric','Off','On','Wins / losses','Difference, 95% interval'],
        [[s['kind'],s['metric'],f'{s["off"]}/{s["n"]}',f'{s["on"]}/{s["n"]}',f'{s["wins"]} / {s["losses"]}',
          f'{100*s["delta"]:+.1f} pp [{100*s["conceptBootstrap95"][0]:+.1f}, {100*s["conceptBootstrap95"][1]:+.1f}]']
         for s in summaries if s['mode']=='hybrid' and s['k']==4 and s['kind']!='all']),
      '<p>Target hit means at least one predeclared supporting passage was retrieved. Complete evidence means all predeclared evidence groups were represented. Equivalent passages may also answer a question, so these are retrieval proxies. Intervals bootstrap 24 concepts; paraphrases are not treated as independent samples.</p>',
      '<h2>Chapter breakdown: complete evidence</h2>',
      table(['Chapter','Wording','Off','On'],[[s['chapter'],s['kind'],f'{s["off"]}/{s["n"]}',f'{s["on"]}/{s["n"]}'] for s in chapter_results])]
if answer_summary:
    s=answer_summary
    body+=['<h2>Answer quality: 24 predetermined casual questions</h2>',
           table(['Metric','Off','On'],[['Both content criteria correct',f'{s["contentCorrect"]["off"]}/24',f'{s["contentCorrect"]["on"]}/24'],
               ['Correct and supported by supplied sources',f'{s["supportedCorrect"]["off"]}/24',f'{s["supportedCorrect"]["on"]}/24'],
               ['Partial or complete, with source support (secondary)',f'{s["supportedPartialOrComplete"]["off"]}/24',f'{s["supportedPartialOrComplete"]["on"]}/24'],
               ['Mean rubric score (0–2)',f'{s["meanScore"]["off"]:.2f}',f'{s["meanScore"]["on"]:.2f}']]),
           f'<p>Supported-answer wins: {s["wins"]}; regressions: {s["losses"]}. Net change {100*s["delta"]:+.1f} percentage points, paired 95% bootstrap interval [{100*s["bootstrap95"][0]:+.1f}, {100*s["bootstrap95"][1]:+.1f}].</p>',
           '<p>The same production source packer, 8,192-token context, 1,024-token answer limit, temperature zero, fixed seed, and no reasoning/rewrite/history were used for both arms. All 48 answers completed normally, fit the token budget, and used four sources. This isolates one-turn answer quality, not the entire application workflow.</p>',
           '<p>The coordinating assistant scored anonymized A/B answers against frozen criteria before decoding the labels. This is independent of the PR author, but not independent human grading. The strict rubric asks for two specific details per concept; a score of one can still be a useful explanation of the question. Partial-or-complete support is a secondary descriptive measure, not a replacement for the primary rubric. All answers and supplied sources are available below and in the raw packet.</p>',
           '<p>The two gains were explaining the tradeoffs of text storage and why sorting must wait for its input. The regression was a less complete explanation of hash indexes for range queries. Enrichment also produced one incorrect decoding-order statement in the bit-packing answer; both arms were already partial on that question, so it does not change the primary complete-answer count.</p>']
body+=['<h2>Amortizing the extra preparation</h2>',
       table(['Lifetime questions','Extra preparation per question'],[[n,f'{extra/n:.2f} s'] for n in [10,100,1000]]),
       '<p>Preparation cost is paid once for unchanged material; reuse can make it acceptable even when it is poor for a one-off document. Break-even depends on the net answer gain and how much time each recovered answer saves. Hypothetical savings are not measured student outcomes.</p>']
if answer_summary:
    body+=[table(['Assumed minutes saved per additional useful answer','Break-even lifetime questions'],
       [[m,('No positive net gain measured' if n is None else math.ceil(n))] for m,n in analysis['breakEvenQueriesAtHypotheticalMinutesSaved'].items()]),
       '<p>These calculations use the observed net gain in fully correct, supported answers. For example, at three minutes saved per extra complete answer, about 97 lifetime questions would recoup the extra 12 minutes. Because the answer interval includes zero or negative improvement, there is no demonstrated finite break-even point. Background preparation can reduce waiting, but it does not establish an accuracy gain.</p>']
body+=['<h2>Search latency and diagnostic arms</h2>',
       table(['Arm','Warm search median','Warm search p95'],[[arm,f'{v["median"]*1000:.1f} ms',f'{v["p95"]*1000:.1f} ms'] for arm,v in search_times.items()]),
       '<p>Search timings exclude query embedding, whose identical vector is reused across arms. No question embedding or model call is mocked: the actual recorded vectors feed the exact production retrieval functions. Two warm repeat passes verified identical retrieval.</p>',
       table(['Mode / K','Wording','Complete evidence off','On','Wins / losses'],[[f'{s["mode"]} / {s["k"]}',s['kind'],f'{s["off"]}/{s["n"]}',f'{s["on"]}/{s["n"]}',f'{s["wins"]} / {s["losses"]}'] for s in summaries if s['metric']=='completeEvidence' and s['kind']!='all']),
       '<p>Using eight sources without enrichment retrieved a target for 33/48 casual questions and complete target evidence for 27/48, versus 32/48 and 22/48 for four enriched sources. This suggests testing a larger source budget as a cheaper alternative. Answer quality at eight sources was not tested; more context may increase answer time or distract the model.</p>',
       '<h2>What “unseen” means here</h2>',
       '<p>The questions were authored before enrichment or retrieval results existed, with zero exact matches in the existing benchmark strings checked locally. They were never exposed to preparation. The PR author’s private question sets were unavailable, and model pretraining overlap cannot be ruled out. These are assistant-authored questions, not actual student submissions. The sample uses four chapters of the same textbook and one local model pair. Intervals describe resampling this small selected sample, not a representative population of students.</p>',
       '<h2>Per-question evidence</h2>']
for c in cases:
    rs={arm:next(r for r in primary if r['id']==c['id'] and r['arm']==arm) for arm in ['off','on']}
    changed=rs['off']['completeEvidence']!=rs['on']['completeEvidence']
    body.append('<details'+(' class="changed"' if changed else '')+'><summary>'+esc(c['id'])+' · '+esc(c['question'])+'</summary>')
    body.append('<p>Frozen required evidence groups: '+esc(c['goldGroups'])+'</p>')
    for arm,r in rs.items():
        body.append('<h3>'+arm.upper()+' · '+('complete evidence' if r['completeEvidence'] else 'incomplete target evidence')+'</h3>')
        body.append('<p>Retrieved IDs: '+esc(r['sourceIds'])+'</p>')
        for source in r['result']['sources']:
            body.append('<details><summary>'+esc(source.get('locator','Source'))+'</summary><pre>'+esc(source.get('context') or source.get('excerpt'))+'</pre></details>')
        if answer_summary and c['generateAnswer']:
            a=next(a for a in answers if a['id']==c['id'] and a['arm']==arm)
            score=next(p[arm] for p in pairs if p['id']==c['id'])
            body.append('<pre class="answer">'+esc(a.get('answer') or a.get('error'))+'</pre><p><strong>Review:</strong> '+esc(score)+'</p>')
    body.append('</details>')
body+=['<h2>Audit and reproduction files</h2><ul>']
for name in ['README.md','artifact-manifest.json','publication-checks.json','verify.py','verify-packing.mjs','new_run.py','summarize.py','protocol.txt','reproduction.txt','validation.json','freeze.json','cases.json','environment.json','integrity.json','novelty-check.json','preparation.json','retrieval.json','retrieval-summary.json','analysis.json','answers.json','blind-review.json','manual-scores.json','blind-key.json','scored-answer-pairs.json','run.py','answers.mjs','make_review.py','build_report.py']:
    if (OUT/name).exists():body.append(f'<li><a href="{name}">{name}</a></li>')
body+=['</ul>']
document='<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Independent PR 157 evaluation</title><style>body{font:16px/1.6 system-ui,sans-serif;color:#23342d;background:#fafbf8;max-width:1080px;margin:45px auto;padding:0 24px}h1{font-size:36px;line-height:1.15;max-width:900px}h2{margin-top:40px}a{color:#185f70}.subtitle{color:#587064}table{border-collapse:collapse;width:100%;margin:20px 0;background:white}th,td{text-align:left;padding:10px 12px;border-bottom:1px solid #d9e3dc}th{background:#e8f0e9}pre{white-space:pre-wrap;font:14px/1.5 ui-monospace,monospace;overflow-wrap:anywhere;background:#f1f4ef;padding:16px}details{margin:10px 0;border:1px solid #d9e3dc;border-radius:7px;padding:12px}summary{cursor:pointer;font-weight:600}.changed{border-left:4px solid #ba7435}.answer{background:#eaf0f5}code{overflow-wrap:anywhere}</style><body>'+''.join(body)+'</body></html>'
(OUT/'report.html').write_text(document)
print(json.dumps(analysis,indent=2))
