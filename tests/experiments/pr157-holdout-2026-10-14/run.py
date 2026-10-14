"""Run the exact PR in disposable profiles using already installed local models.

PYTHON run.py prepare|search --source PATH --work PATH
Requires numpy/faiss and local Ollama; never pulls models or touches the app profile.
"""
import argparse
import contextlib
import hashlib
import io
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

OUT = Path(__file__).resolve().parent
ARGS = argparse.ArgumentParser()
ARGS.add_argument('stage', choices=['prepare','search'])
ARGS.add_argument('--source', type=Path, required=True)
ARGS.add_argument('--work', type=Path, required=True)
args = ARGS.parse_args()
args.work.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(args.source.resolve()))
from python_engine import tokensmith_engine as engine, tokensmith_store as store
from python_engine import tokensmith_preparation as prep
import numpy as np

def write(name, value):
    (OUT/name).write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

freeze = json.loads((OUT/'freeze.json').read_text())
for name, expected in freeze['hashes'].items():
    assert sha(OUT/name) == expected, f'Frozen input changed: {name}'
assert store.faiss is not None, 'FAISS required; no keyword fallback permitted'
store.faiss.omp_set_num_threads(1)

EMBEDDER = {'id':'ollama:nomic-embed-text', 'engine':'ollama', 'name':'nomic-embed-text',
            'role':'embedder', 'ollamaModelName':'nomic-embed-text:latest', 'contextLength':2048}
GENERATOR = {'id':'ollama:gemma4:e4b', 'engine':'ollama', 'name':'gemma4:e4b',
             'role':'generator', 'ollamaModelName':'gemma4:e4b', 'contextLength':8192}
corpus = (OUT/'corpus.tokensmith.md').resolve()
original_urlopen = urllib.request.urlopen

def api(route):
    with original_urlopen('http://127.0.0.1:11434/api/'+route, timeout=20) as response:
        return json.load(response)

def profile(arm):
    return args.work/arm

def hardware_metadata():
    # Portability-only change after the measured run. Original script is retained.
    if platform.system() == 'Darwin':
        return (subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip(),
                int(subprocess.check_output(['sysctl','-n','hw.memsize'],text=True)))
    cpu = platform.processor() or platform.machine()
    memory = None
    try:
        memory = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
    except (AttributeError, ValueError, OSError):
        pass
    return cpu, memory

def prepare():
    cpu, memory = hardware_metadata()
    environment = {'startedAt':datetime.now(timezone.utc).isoformat(), 'platform':platform.platform(),
                   'python':sys.version, 'numpy':np.__version__, 'faiss':store.faiss.__version__,
                   'prHead':freeze['prHead'], 'ollamaVersion':api('version'),
                   'initialResidency':api('ps'), 'models':api('tags'),
                   'cpu':cpu,
                   'memoryBytes':memory,
                   'sourceHashes':{str(p.relative_to(args.source)):sha(p) for p in sorted((args.source/'python_engine').glob('*.py'))},
                   'corpusPath':str(corpus), 'work':str(args.work), 'generator':GENERATOR, 'embedder':EMBEDDER}
    write('environment.json', environment)
    timings = []
    for arm in ['off','on']:
        assert not (profile(arm)/'tokensmith.sqlite').exists(), 'Cold profile must be empty'
        for run in ['cold','cached']:
            started = time.perf_counter(); events=[]; calls=[]; last_progress=[None,-100]
            def progress(request, event):
                events.append({'elapsedSeconds':time.perf_counter()-started, **event})
                done=event.get('processedEmbeddings',0)
                if event.get('phase') != last_progress[0] or done-last_progress[1]>=50:
                    print(arm,run,event.get('phase'),event.get('percent'),event.get('message'),flush=True)
                    last_progress[:]=[event.get('phase'),done]
            @contextlib.contextmanager
            def observed_open(request, *a, **kw):
                req=json.loads(request.data) if hasattr(request,'data') and request.data else {}
                start=time.perf_counter()
                with original_urlopen(request,*a,**kw) as response:
                    data=response.read()
                result=json.loads(data)
                record={'seconds':time.perf_counter()-start,'url':getattr(request,'full_url',str(request)),
                        'model':req.get('model'),'request':{k:v for k,v in req.items() if k not in ('input','prompt')},
                        'inputChars':len(str(req.get('input',req.get('prompt','')))),
                        'response':{k:v for k,v in result.items() if k not in ('embeddings','embedding')}}
                calls.append(record)
                with (OUT/f'{arm}-{run}-calls.jsonl').open('a') as log:
                    log.write(json.dumps(record,ensure_ascii=False)+'\n')
                if '/api/chat' in record['url'] and sum('/api/chat' in x['url'] for x in calls)%20==0:
                    print(arm,run,'descriptions',sum('/api/chat' in x['url'] for x in calls),
                          'elapsed',round(time.perf_counter()-started,1),flush=True)
                yield io.BytesIO(data)
            payload={'userDataPath':str(profile(arm)),'path':str(corpus),'materialId':'holdout',
                     'title':'Independent chapter sample','model':EMBEDDER,
                     'preparation':{'mode':'basic','instructions':'','documentInstructions':{},'plainLanguageTerms':arm=='on'},
                     'preparationModel':GENERATOR if arm=='on' else None}
            with patch.object(engine,'send_progress',progress), patch.object(urllib.request,'urlopen',observed_open):
                material=engine.index_material(payload)['material']
            elapsed=time.perf_counter()-started
            dumped=store.dump_index(str(profile(arm)))
            chunks=dumped['chunks']
            descriptions=[c for c in calls if '/api/chat' in c['url']]
            embeddings=[c for c in calls if '/api/embed' in c['url']]
            timing={'arm':arm,'run':run,'totalSeconds':elapsed,'generatorSeconds':sum(c['seconds'] for c in descriptions),
                    'embeddingSeconds':sum(c['seconds'] for c in embeddings),'generatorCalls':len(descriptions),
                    'embeddingCalls':len(embeddings),'loadSeconds':sum(c['response'].get('load_duration',0)/1e9 for c in calls),
                    'chunks':len(chunks),'chunksWithTerms':sum(bool(c.get('keywords')) for c in chunks),
                    'material':material,'progress':events}
            timings.append(timing);write('preparation.json',timings)
            write(f'{arm}-stored-chunks.json',chunks)
            assert run!='cached' or len(descriptions)==0, 'Cached enrichment unexpectedly regenerated'
            print('COMPLETE',arm,run,json.dumps({k:v for k,v in timing.items() if k not in ['material','progress']}),flush=True)
    before=json.loads((OUT/'off-stored-chunks.json').read_text())
    after=json.loads((OUT/'on-stored-chunks.json').read_text())
    assert [c['chunk_text'] for c in before] == [c['chunk_text'] for c in after], 'Passages changed'
    write('integrity.json',{'sourceTextByteIdentical':True,'chunkCountEqual':len(before)==len(after),
                          'chunks':len(before),'coldAndCachedComplete':True})

def search():
    cases=json.loads((OUT/'cases.json').read_text())
    embedding_key, embed, reason=engine.resolve_embedding_provider_from_spec(EMBEDDER)
    assert embed is not None and not reason, reason
    vectors={};query_times=[]
    for c in cases:
        start=time.perf_counter();vectors[c['id']]=embed(c['question']);query_times.append(time.perf_counter()-start)
    write('query-embeddings.json',vectors);write('query-embedding-times.json',query_times)
    materials={arm:store.list_materials(str(profile(arm))) for arm in ['off','on']}
    assert all(len(m)==1 for m in materials.values())
    results=[];timing_repeats=[]
    conditions=[('hybrid',4),('hybrid',8),('keyword',4),('vector',4)]
    for ci,c in enumerate(cases):
        for mode,k in conditions:
            for arm in (['off','on'] if ci%2==0 else ['on','off']):
                with patch.object(engine,'resolve_embedding_provider_for_key',return_value=(lambda _:vectors[c['id']],None)):
                    start=time.perf_counter()
                    result=engine.search_library({'userDataPath':str(profile(arm)),'query':c['question'],
                        'materials':materials[arm],'searchMode':mode,'limit':k})
                    elapsed=time.perf_counter()-start
                assert not result['reason'], result['reason']
                ids=set(x for s in result['sources'] for x in (s.get('sourceChunkIds') or [s.get('tokensmithChunkId')]) if x)
                hits=[bool(set(g)&ids) for g in c['goldGroups']]
                results.append({'id':c['id'],'concept':c['concept'],'kind':c['kind'],'chapter':c['chapter'],
                    'arm':arm,'mode':mode,'k':k,'seconds':elapsed,'goldGroupHits':hits,
                    'targetHit':any(hits),'completeEvidence':all(hits),'sourceIds':sorted(ids),
                    'question':c['question'],'result':result})
        if ci%12==0:
            print('SEARCH',ci+1,'/',len(cases),flush=True);write('retrieval.json',results)
    write('retrieval.json',results)
    # Two additional interleaved passes for descriptive warm timing and determinism.
    for repeat in range(2):
        order=list(range(len(cases)));random.Random(157+repeat).shuffle(order)
        for ci in order:
            c=cases[ci]
            for arm in (['off','on'] if (ci+repeat)%2==0 else ['on','off']):
                with patch.object(engine,'resolve_embedding_provider_for_key',return_value=(lambda _:vectors[c['id']],None)):
                    start=time.perf_counter();result=engine.search_library({'userDataPath':str(profile(arm)),
                        'query':c['question'],'materials':materials[arm],'searchMode':'hybrid','limit':4})
                    elapsed=time.perf_counter()-start
                ids=sorted(set(x for s in result['sources'] for x in (s.get('sourceChunkIds') or [s.get('tokensmithChunkId')]) if x))
                original=next(r for r in results if r['id']==c['id'] and r['arm']==arm and r['mode']=='hybrid' and r['k']==4)
                assert ids==original['sourceIds'],'Nondeterministic retrieval'
                timing_repeats.append({'id':c['id'],'arm':arm,'repeat':repeat+1,'seconds':elapsed})
    write('search-timings.json',timing_repeats)
    summaries=[]
    for mode,k in conditions:
        for kind in ['casual','technical','all']:
            subset=[r for r in results if r['mode']==mode and r['k']==k and (kind=='all' or r['kind'].startswith(kind))]
            for metric in ['targetHit','completeEvidence']:
                pair={arm:{r['id']:r[metric] for r in subset if r['arm']==arm} for arm in ['off','on']}
                off,on=pair['off'],pair['on']
                summaries.append({'mode':mode,'k':k,'kind':kind,'metric':metric,'n':len(off),
                    'off':sum(off.values()),'on':sum(on.values()),
                    'wins':sum(on[i] and not off[i] for i in off),'losses':sum(off[i] and not on[i] for i in off)})
    write('retrieval-summary.json',summaries)
    print(json.dumps([s for s in summaries if s['mode']=='hybrid' and s['k']==4],indent=2),flush=True)

prepare() if args.stage=='prepare' else search()
