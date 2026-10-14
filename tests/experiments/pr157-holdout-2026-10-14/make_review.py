"""Produce opaque A/B packets as completed answer pairs become available."""
import json
import secrets
from pathlib import Path

out=Path(__file__).resolve().parent
read=lambda name:json.loads((out/name).read_text())
cases=[c for c in read('cases.json') if c['generateAnswer']]
if (out/'blind-key.json').exists():
    key=read('blind-key.json')
else:
    key={}
    for c in cases:
        flip=secrets.randbelow(2)
        for i,arm in enumerate(['off','on']): key[c['id']+'.'+('B' if i^flip else 'A')]=arm
    (out/'blind-key.json').write_text(json.dumps(key,indent=2)+'\n')
answers=read('answers.json');packet=[]
for c in cases:
    pair=[a for a in answers if a['id']==c['id']]
    if len(pair)!=2: continue
    for label in ['A','B']:
        review_id=c['id']+'.'+label
        a=next(a for a in pair if a['arm']==key[review_id])
        packet.append({'reviewId':review_id,'question':c['question'],'criteria':c['answerCriteria'],
            'goldGroups':c['goldGroups'],'answer':a.get('answer'),'error':a.get('error'),
            'sources':[{'title':s.get('title'),'locator':s.get('locator'),'ids':s.get('sourceChunkIds'),
                        'text':s.get('context') or s.get('excerpt')} for s in a.get('sources',[])]})
(out/'blind-review.json').write_text(json.dumps(packet,indent=2,ensure_ascii=False)+'\n')
print(len(packet)//2,'complete blinded pairs')
