"""Source-bound evidence labels for the unified hybrid accuracy benchmark.

Required claims are ANDed; each alternative must be complete in the actual
returned context. These are development annotations, not held-out expert gold.
Run via `npm run test:benchmark`; no separate model, index, or scoring command.
"""
import hashlib
import itertools
import json
from pathlib import Path

from python_engine import tokensmith_engine as engine

ROOT = Path(__file__).resolve().parents[2]
CASES = Path(__file__).with_name('buzzdb_retrieval_cases.json')


def normalize(text):
    return ' '.join(text.casefold().split())


def validate_review(review, source_chunks):
    """Validate review provenance and every quoted citation against original sources."""
    cases = {case['id']: case for case in review['cases']}
    if len(review['cases']) != len(cases):
        raise ValueError('Duplicate reviewed case.')
    for case in review['cases']:
        if not case['question'].strip():
            raise ValueError(f"Empty question: {case['id']}")
        groups = case['groups']
        if not groups or len({g['id'] for g in groups}) != len(groups):
            raise ValueError(f"Missing or duplicate evidence groups: {case['id']}")
        required = [group for group in groups if group['required']]
        if case['answerability'] != 'answerable' or not required:
            raise ValueError(f"Retrieval coverage cases need required, answerable facts: {case['id']}")
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
    return {'complete': bool(required) and all(d['hit'] for d in required),
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
    # Variants inherit evidence, rather than keeping a second copy that can drift.
    originals = {case['id']: case for case in data['cases'] if not case.get('pairedWith')}
    for case in data['cases']:
        if case.get('pairedWith'):
            original = originals[case['pairedWith']]
            case['question'] = case['background'] + '\n\n' + original['question']
            case['answerability'] = original['answerability']
            case['groups'] = original['groups']
            case['evaluationProfile'] = original['evaluationProfile']
    fixture = ROOT / data['source']['path']
    if hashlib.sha256(fixture.read_bytes()).hexdigest() != data['source']['sha256']:
        raise ValueError('Book fixture changed; review the labels before scoring.')
    chunks = engine.chunk_tokensmith_markdown(fixture.read_text(encoding='utf-8'))
    sources = {(data['source']['id'], c['tokensmithChunkId']): c['text'] for c in chunks}
    if len(chunks) != len(sources) or len(sources) != data['source']['chunks']:
        raise ValueError('Missing or duplicate source chunks.')
    validate_review(data, sources)
    cases = {c['id']: c for c in data['cases']}
    for c in cases.values():
        supports = minimum_supports(c)
        structure = c['evaluationProfile']['evidenceStructure']
        if (min(map(len, supports)) != structure['minimumKnownChunks'] or
            min(len({cid.split('.')[0] for cid in s}) for s in supports) != structure['minimumKnownChapters'] or
            sum(g['required'] for g in c['groups']) != structure['requiredFacts']):
            raise ValueError(f"Stale evidence classification: {c['id']}")
    return cases, sources
