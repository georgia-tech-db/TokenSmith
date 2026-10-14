"""Audit the published evaluation without model calls or third-party packages."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def read(name):
    return json.loads((ROOT / name).read_text())

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def verify_source(source):
    for name, expected in read('environment.json')['sourceHashes'].items():
        assert digest(source / name) == expected, f'PR engine differs: {name}'
    assert digest(source / 'src/main/engine/study-chat-format.ts') == read('validation.json')['answerPackerSha256'], 'Answer packer differs'

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, help='Also verify the PR source snapshot')
    args = parser.parse_args()
    manifest = read('artifact-manifest.json')
    for name, entry in manifest['files'].items():
        path = ROOT / name
        assert path.is_file() and path.stat().st_size == entry['bytes'], f'Missing or changed: {name}'
        assert digest(path) == entry['sha256'], f'Checksum mismatch: {name}'
    for name, expected in read('freeze.json')['hashes'].items():
        assert digest(ROOT / name) == expected, f'Frozen input changed: {name}'
    if args.source:
        verify_source(args.source.resolve())
    cases, answers, scores = read('cases.json'), read('answers.json'), read('manual-scores.json')
    assert len(cases) == 72 and len({c['concept'] for c in cases}) == 24
    assert len(answers) == len(scores) == 48
    assert len({(a['id'], a['arm']) for a in answers}) == 48
    assert all(not a.get('error') and a['calls'][-1]['response']['done_reason'] == 'stop' for a in answers)
    assert all(len(a['sources']) == 4 and a['budget']['truncatedSourceCount'] == 0 for a in answers)
    assert all(a['calls'][-1]['response']['prompt_eval_count'] + a['budget']['answerReserveTokens'] + a['budget']['safetyMarginTokens'] <= 8192 for a in answers)
    off, on = read('off-stored-chunks.json'), read('on-stored-chunks.json')
    assert len(off) == len(on) == 416
    assert [c['chunk_text'] for c in off] == [c['chunk_text'] for c in on]
    assert sum(bool(c.get('keywords')) for c in on) == 251
    key = read('blind-key.json')
    correct = {arm: sum(s['score'] == 2 and s['supported'] for s in scores if key[s['reviewId']] == arm) for arm in ['off', 'on']}
    assert correct == read('answer-summary.json')['supportedCorrect'] == {'off': 11, 'on': 12}
    retrieval = read('retrieval.json')
    assert len(retrieval) == 576
    casual = [r for r in retrieval if r['mode'] == 'hybrid' and r['k'] == 4 and r['kind'].startswith('casual')]
    hit = {arm: sum(r['targetHit'] for r in casual if r['arm'] == arm) for arm in ['off', 'on']}
    assert hit == {'off': 28, 'on': 32}
    preparation = read('preparation.json')
    assert all(r['generatorCalls'] == r['embeddingCalls'] == 0 for r in preparation if r['run'] == 'cached')
    print(f"PASS: {len(manifest['files'])} published files match their checksums.")
    print('PASS: frozen inputs, 72 questions, 24 concepts, 416 original passages, 48 answers.')
    print('PASS: primary results recomputed: retrieval 28 -> 32 / 48; complete supported answers 11 -> 12 / 24.')
    if args.source:
        print('PASS: PR engine and answer-packer hashes match the tested snapshot.')

if __name__ == '__main__':
    main()
