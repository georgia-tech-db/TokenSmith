"""Compute fresh paired statistics from a rerun; no hard-coded result narrative."""
import argparse
import json
import statistics
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--directory', type=Path, default=Path(__file__).resolve().parent)
args = parser.parse_args()
root = args.directory.resolve()
read = lambda name: json.loads((root/name).read_text())

def paired(rows, metric):
    groups = {arm: {r['id']: r for r in rows if r['arm'] == arm} for arm in ['off', 'on']}
    assert groups['off'].keys() == groups['on'].keys()
    deltas = {i: int(metric(groups['on'][i])) - int(metric(groups['off'][i])) for i in groups['off']}
    concepts = {r['concept'] for r in rows}
    cluster_means = [statistics.mean(deltas[i] for i, r in groups['off'].items() if r['concept'] == c) for c in sorted(concepts)]
    samples = np.random.default_rng(157).choice(cluster_means, (20000, len(concepts)), replace=True).mean(axis=1)
    return {'n': len(deltas), 'concepts': len(concepts),
            **{arm: sum(bool(metric(r)) for r in groups[arm].values()) for arm in ['off', 'on']},
            'wins': list(deltas.values()).count(1), 'losses': list(deltas.values()).count(-1),
            'delta': statistics.mean(deltas.values()),
            'conceptBootstrap95': [float(x) for x in np.quantile(samples, [.025, .975])]}

retrieval = read('retrieval.json')
summary = {'retrieval': []}
for mode, k in sorted({(r['mode'], r['k']) for r in retrieval}):
    for kind in ['casual', 'technical', 'all']:
        rows = [r for r in retrieval if r['mode'] == mode and r['k'] == k and (kind == 'all' or r['kind'].startswith(kind))]
        for metric in ['targetHit', 'completeEvidence']:
            summary['retrieval'].append({'mode': mode, 'k': k, 'kind': kind, 'metric': metric,
                                         **paired(rows, lambda r: r[metric])})
prep = read('preparation.json')
summary['preparationSeconds'] = {arm: {r['run']: r['totalSeconds'] for r in prep if r['arm'] == arm} for arm in ['off', 'on']}
summary['extraInitialSeconds'] = summary['preparationSeconds']['on']['cold'] - summary['preparationSeconds']['off']['cold']
if (root/'manual-scores.json').exists():
    key = read('blind-key.json')
    concepts = {c['id']: c['concept'] for c in read('cases.json')}
    rows = []
    for score in read('manual-scores.json'):
        ident = score['reviewId'].rsplit('.', 1)[0]
        rows.append({**score, 'id': ident, 'concept': concepts[ident], 'arm': key[score['reviewId']]})
    expected = {c['id'] for c in read('cases.json') if c['generateAnswer']}
    assert len(rows) == 2*len(expected) and {(r['id'], r['arm']) for r in rows} == {(i, arm) for i in expected for arm in ['off', 'on']}
    summary['supportedCompleteAnswers'] = paired(rows, lambda r: r['score'] == 2 and r['supported'])
    summary['supportedPartialOrCompleteAnswers'] = paired(rows, lambda r: r['score'] >= 1 and r['supported'])
else:
    summary['answerScoring'] = 'Pending: finish blind manual-scores.json before decoding labels.'
output = root/'rerun-summary.json'
output.write_text(json.dumps(summary, indent=2)+'\n')
print('Saved:', output)
print(json.dumps({k: v for k, v in summary.items() if k != 'retrieval'}, indent=2))
