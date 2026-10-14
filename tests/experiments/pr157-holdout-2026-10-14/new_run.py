"""Create a fresh reproduction directory; never overwrite the published results."""
import argparse
import json
import shlex
import shutil
import sys
import tempfile
from pathlib import Path
from verify import ROOT, digest, read, verify_source

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', required=True, type=Path, help='Checkout containing the tested PR code')
parser.add_argument('--name', default='pr157-holdout-rerun', help='New folder name under tests/experiments')
args = parser.parse_args()
source = args.source.resolve()
assert args.name not in {'.', '..'} and Path(args.name).name == args.name and '/' not in args.name and '\\' not in args.name
verify_source(source)
for name, expected in read('freeze.json')['hashes'].items():
    assert digest(ROOT / name) == expected, f'Frozen input changed: {name}'
target = source / 'tests/experiments' / args.name
target.mkdir(exist_ok=False)
names = list(read('freeze.json')['hashes']) + [
    'freeze.json', 'protocol.txt', 'run.py', 'answers.mjs', 'ts-module-loader.mjs', 'make_review.py', 'summarize.py',
]
for name in names:
    shutil.copy2(ROOT / name, target / name)
shutil.copy2(ROOT / 'environment.json', target / 'reference-environment.json')
work = Path(tempfile.mkdtemp(prefix='tokensmith-pr157-rerun-'))
(target / 'rerun-setup.json').write_text(json.dumps({'source': str(source), 'work': str(work)}, indent=2) + '\n')
quote = shlex.quote
print('Created:', target)
print('Fresh isolated indexes:', work)
print('No model calls have been made. Confirm the model digests in reference-environment.json, then run:')
for stage in ['prepare', 'search']:
    print(' '.join(quote(str(x)) for x in [sys.executable, target/'run.py', stage, '--source', source, '--work', work]))
print(' '.join(quote(str(x)) for x in ['node', target/'answers.mjs', '--source', source]))
print(' '.join(quote(str(x)) for x in [sys.executable, target/'make_review.py']))
print('Grade blind-review.json against its frozen criteria before reading blind-key.json.')
