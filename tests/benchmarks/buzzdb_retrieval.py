"""Reviewed BuzzDB retrieval cases, using the existing book fixture.

Validate without indexing or model calls:
  app_runtime/python/bin/python tests/benchmarks/buzzdb_retrieval.py validate

Run live retrieval with the original benchmark's model, cache and index setup:
  npm run test:retrieval-benchmark -- --out tmp/reviewed-retrieval.json
  npm run test:retrieval-benchmark -- --case same_bitpacked_slice_alignment

Score saved retrievals:
  app_runtime/python/bin/python tests/benchmarks/buzzdb_retrieval.py score results.json

Input: {"cases": [{"id": case_id, "mode": "vector", "k": 4, "sources":
[{"document": "buzzdb", "chunk_id": "ch15.055", "text": actual_context}]}]}.
Use the question alone as retriever input; never embed labels or metadata.
Any retriever/model can supply this format. Scores measure known-evidence
coverage; unjudged passages are not negative qrels. Partial runs are identified.

The live runner shares the original benchmark's pinned Nomic model and temporary
index. It defaults to Vector/Hybrid at 4/8 sources. Evidence misses are scores;
model/setup/search errors fail the run. No answers are generated.

Build a reusable book cache once:
  npm run build:embedding-benchmark-cache -- tmp/buzzdb-nomic-cache.npz
Set TOKENSMITH_BENCHMARK_EMBEDDINGS_PATH to that file when running; otherwise the
book is embedded again. Each search embeds its question afresh. Model, source
text, chunk order and embedding-input metadata are validated.

Default Ollama version: tests/benchmarks/embedding-model.json. For an explicitly
chosen different installed version, set TOKENSMITH_BENCHMARK_OLLAMA_VERSION for
both cache creation and execution. The selected version is checked and recorded;
the Nomic digest remains pinned. Existing result files are preserved.

Use --modes hybrid --k 4 or repeat --case ID for selected runs. These reviewed
labels are development annotations, not independently annotated held-out gold.
"""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from python_engine import tokensmith_engine as engine

CASES = Path(__file__).with_name('buzzdb_retrieval_cases.json')


def normalize(text):
    return ' '.join(text.casefold().split())


def validate_review(definition, review, source_chunks):
    """Validate review provenance and every quoted citation against original sources."""
    cases = {case['id']: case for case in definition['cases']}
    if len(review['cases']) != len(cases) or {case['id'] for case in review['cases']} != set(cases):
        raise ValueError('Review must cover every original question exactly once.')
    if len({case['id'] for case in review['cases']}) != len(cases):
        raise ValueError('Duplicate reviewed case.')
    for case in review['cases']:
        original = cases[case['id']]
        digest = hashlib.sha256(original['question'].encode()).hexdigest()
        if digest != case['questionSha256'] or case['answerability'] != original['answerability']:
            raise ValueError(f"Question or answerability changed: {case['id']}")
        groups = case['groups']
        if not groups or len({g['id'] for g in groups}) != len(groups):
            raise ValueError(f"Missing or duplicate evidence groups: {case['id']}")
        required = [group for group in groups if group['required']]
        if (case['answerability'] == 'answerable') != bool(required):
            raise ValueError(f"Insufficient cases must not receive a retrieval-success score: {case['id']}")
        for group in groups:
            if not group['claim'] or not group['alternatives']:
                raise ValueError('Each claim needs at least one sufficient alternative.')
            if len({a['id'] for a in group['alternatives']}) != len(group['alternatives']):
                raise ValueError('Duplicate alternative ID.')
            for alternative in group['alternatives']:
                if not alternative['passages']:
                    raise ValueError('An empty alternative cannot prove a claim.')
                for passage in alternative['passages']:
                    text = source_chunks[(passage['sourceId'], passage['chunkId'])]
                    if not passage['quotes'] or any(not quote.strip() for quote in passage['quotes']):
                        raise ValueError('Each citation needs nonempty literal evidence.')
                    for quote in passage['quotes']:
                        if normalize(quote) not in normalize(text):
                            raise ValueError(f"Missing literal evidence: {case['id']} / {passage['chunkId']} / {quote}")


def score_evidence(reviewed_case, selected):
    """All required claims; OR alternatives; AND source-bound passages and quotes."""
    selected_by_key = {}
    for candidate in selected:
        key = candidate['document'], candidate['chunk_id']
        selected_by_key.setdefault(key, []).append(candidate['text'])
    decisions = []
    for group in reviewed_case['groups']:
        matched = []
        for alternative in group['alternatives']:
            valid = True
            for passage in alternative['passages']:
                texts = selected_by_key.get((passage['sourceId'], passage['chunkId']), [])
                # Every span must survive together in this specific passage's context.
                if not any(text.strip() and all(normalize(q) in normalize(text) for q in passage['quotes']) for text in texts):
                    valid = False
                    break
            if valid:
                matched.append(alternative['id'])
        decisions.append({'id': group['id'], 'required': group['required'], 'hit': bool(matched),
                          'matched_alternatives': matched})
    required = [decision for decision in decisions if decision['required']]
    scored = reviewed_case['answerability'] == 'answerable'
    return {'scored': scored, 'complete': all(d['hit'] for d in required) if scored else None,
            'group_hits': [d['hit'] for d in required],
            'group_recall': sum(d['hit'] for d in required) / len(required) if scored else None,
            'groups': decisions}

def minimum_supports(case):
    supports = {frozenset()}
    for group in case['groups']:
        if group['required']:
            choices = {frozenset(p['chunkId'] for p in a['passages']) for a in group['alternatives']}
            unions = {a | b for a, b in itertools.product(supports, choices)}
            supports = {s for s in unions if not any(t < s for t in unions)}
    return supports


def load():
    data = json.loads(CASES.read_text(encoding='utf-8'))
    fixture = ROOT / data['source']['path']
    if hashlib.sha256(fixture.read_bytes()).hexdigest() != data['source']['sha256']:
        raise ValueError('Book fixture changed; review the labels before scoring.')
    chunks = engine.chunk_tokensmith_markdown(fixture.read_text(encoding='utf-8'))
    sources = {(data['source']['id'], c['tokensmithChunkId']): c['text'] for c in chunks}
    if len(chunks) != len(sources) or len(sources) != data['source']['chunks']:
        raise ValueError('Missing or duplicate source chunks.')
    validate_review(data, data, sources)
    cases = {c['id']: c for c in data['cases']}
    for c in cases.values():
        supports = minimum_supports(c)
        structure = c['evaluationProfile']['evidenceStructure']
        if (min(map(len, supports)) != structure['minimumKnownChunks'] or
            min(len({cid.split('.')[0] for cid in s}) for s in supports) != structure['minimumKnownChapters'] or
            sum(g['required'] for g in c['groups']) != structure['requiredFacts']):
            raise ValueError(f"Stale evidence classification: {c['id']}")
        if c.get('pairedWith'):
            original = cases[c['pairedWith']]
            if c['groups'] != original['groups'] or c['question'] != c['background']+'\n\n'+original['question']:
                raise ValueError('Background pair changed its question or evidence.')
    return cases, sources


def score(cases, rows):
    if not rows:
        raise ValueError('No retrieval results to score.')
    seen, decisions = set(), []
    for row in rows:
        key = row['id'], row['mode'], row['k']
        if key in seen:
            raise ValueError(f'Duplicate result: {key}')
        seen.add(key)
        case = cases[row['id']]
        decisions.append({'id':row['id'], 'mode':row['mode'], 'k':row['k'],
                          'metrics':score_evidence(case, row['sources'])})
    slices = {
        'base_all': lambda c: True,
        'named_concept': lambda c: c['evaluationProfile']['queryForm']=='named_concept',
        'described_behavior': lambda c: c['evaluationProfile']['queryForm']=='described_behavior',
        'technical_scenario': lambda c: c['evaluationProfile']['queryForm']=='technical_scenario',
        'discrimination': lambda c: c['evaluationProfile']['distinction'] is not None,
        'single_chunk': lambda c: c['evaluationProfile']['evidenceStructure']['minimumKnownChunks']==1,
        'multi_chunk': lambda c: c['evaluationProfile']['evidenceStructure']['minimumKnownChunks']>1,
        'same_chapter_multi_chunk': lambda c: c['evaluationProfile']['evidenceStructure']['minimumKnownChunks']>1 and c['evaluationProfile']['evidenceStructure']['minimumKnownChapters']==1,
        'cross_chapter': lambda c: c['evaluationProfile']['evidenceStructure']['minimumKnownChapters']>1,
        'code_required': lambda c: 'code' in c['evaluationProfile']['evidenceFormats']['necessaryUnderCurrentLabels'],
        'table_required': lambda c: 'table' in c['evaluationProfile']['evidenceFormats']['necessaryUnderCurrentLabels'],
    }
    groups, missing = [], []
    for mode,k in sorted({(r['mode'],r['k']) for r in decisions}):
        selected = [r for r in decisions if r['mode']==mode and r['k']==k]
        missing.append({'mode':mode,'k':k,'missingIds':sorted(set(cases)-{r['id'] for r in selected})})
        for name, predicate in slices.items():
            eligible = {cid for cid,c in cases.items() if not c.get('pairedWith') and predicate(c)}
            present = [r for r in selected if r['id'] in eligible]
            groups.append({'slice':name,'mode':mode,'k':k,'expectedQuestions':len(eligible),
                           'evaluatedQuestions':len(present),'complete':sum(r['metrics']['complete'] for r in present)})
    lookup = {(r['id'],r['mode'],r['k']):r for r in decisions}
    pairs = []
    for row in decisions:
        original = cases[row['id']].get('pairedWith')
        if original and (original,row['mode'],row['k']) in lookup:
            before = lookup[original,row['mode'],row['k']]
            pairs.append({'id':row['id'],'originalId':original,'mode':row['mode'],'k':row['k'],
                          'originalComplete':before['metrics']['complete'],
                          'backgroundComplete':row['metrics']['complete']})
    return {'scope':'Known-evidence coverage, not answer accuracy or precision.',
            'datasetSha256':hashlib.sha256(CASES.read_bytes()).hexdigest(),
            'missingCasesByConfiguration':missing,'groups':groups,'backgroundPairs':pairs,'cases':decisions}


def run(cases, modes, limits):
    # Share the original benchmark's pinned model checks, passage-cache
    # validation, temporary index, live embeddings and fallback detection.
    from tests.benchmarks import test_buzzdb_embeddings as live
    fixture = live.BuzzDBEmbeddingBenchmarkTests
    rows = []
    try:
        with patch.dict(os.environ, {'TOKENSMITH_RUN_EMBEDDING_BENCHMARK':'1'}):
            fixture.setUpClass()
        client = fixture()
        for case in cases.values():
            for mode in modes:
                for k in limits:
                    started = time.perf_counter()
                    row = {'id':case['id'],'question':case['question'],'mode':mode,'k':k,'sources':[]}
                    before = client.measurements['fresh_query_embedding_seconds']
                    try:
                        result = client.retrieve_sources(case['question'], k, mode)
                        for rank, source in enumerate(result['sources'], 1):
                            ids = source.get('sourceChunkIds') or [source['tokensmithChunkId']]
                            row['sources'].extend({'document':'buzzdb','chunk_id':cid,
                                                   'text':source['context'],'rank':rank} for cid in ids)
                    except Exception as error:
                        row['error'] = str(error)
                    row['searchSecondsIncludingQueryEmbedding'] = time.perf_counter()-started
                    row['queryEmbeddingSeconds'] = client.measurements['fresh_query_embedding_seconds']-before
                    rows.append(row)
            print(f"Retrieved {case['id']}", file=sys.stderr, flush=True)
        output = score(cases, rows)
        output['retrievals'] = rows
        output['runtimeErrors'] = [{'id':r['id'],'mode':r['mode'],'k':r['k'],'error':r['error']}
                                   for r in rows if 'error' in r]
        output['controls'] = {'model':live.MODEL,'searchModes':modes,'limits':limits,
            'documentEmbeddingCache':os.environ.get('TOKENSMITH_BENCHMARK_EMBEDDINGS_PATH'),
            'embeddingCacheMetadata':fixture.bundle['metadata'],
            'engineSha256':hashlib.sha256(Path(engine.__file__).read_bytes()).hexdigest(),
            'storeSha256':hashlib.sha256(Path(live.store.__file__).read_bytes()).hexdigest(),
            'measurements':client.measurements,'freshQueryEmbeddingPerSearch':True,
            'index':'Temporary benchmark index; no student library changes.',
            'generationCalls':0,'exitPolicy':'Evidence misses are reported scores; setup/retrieval errors fail the run.'}
        return output
    finally:
        fixture.doClassCleanups()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command',choices=['validate','score','run'])
    parser.add_argument('results',type=Path,nargs='?')
    parser.add_argument('--out',type=Path)
    parser.add_argument('--modes',nargs='+',choices=['vector','hybrid'],default=['vector','hybrid'])
    parser.add_argument('--k',nargs='+',type=int,default=[4,8])
    parser.add_argument('--case',dest='case_ids',action='append',help='Run selected question IDs only (repeatable).')
    args=parser.parse_args()
    cases,sources=load()
    if args.out and args.out.exists():
        parser.error('Choose a new output file; existing benchmark results are preserved.')
    if any(k<1 for k in args.k) or len(set(args.k))!=len(args.k) or len(set(args.modes))!=len(args.modes):
        parser.error('Limits must be positive and mode/limit lists must not contain duplicates.')
    if args.command=='validate':
        output={'questions':len(cases),'baseQuestions':sum(not c.get('pairedWith') for c in cases.values()),
                'sourceChunks':len(sources),'sourceBoundQuotesValidated':True,'modelCalls':0}
    elif args.command=='score':
        if args.results is None:
            parser.error('score requires a results JSON file')
        saved=json.loads(args.results.read_text(encoding='utf-8'))
        output=score(cases,saved.get('retrievals',saved['cases']))
    else:
        if args.case_ids:
            unknown=set(args.case_ids)-set(cases)
            if unknown:
                parser.error('Unknown question IDs: '+', '.join(sorted(unknown)))
            cases={cid:case for cid,case in cases.items() if cid in args.case_ids}
        output=run(cases,args.modes,args.k)
    text=json.dumps(output,ensure_ascii=False,indent=2)+'\n'
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        with args.out.open('x',encoding='utf-8') as stream:
            stream.write(text)
    else:
        print(text,end='')
    if args.command=='run':
        for group in output['groups']:
            print(f"{group['slice']} {group['mode']}@{group['k']}: {group['complete']}/{group['evaluatedQuestions']}",file=sys.stderr)
        raise SystemExit(1 if output['runtimeErrors'] else 0)
