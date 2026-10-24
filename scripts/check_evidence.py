"""Check preserved reports against their recorded revision or this checkout."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


def verify(root, entry, *, current=False):
    path = root / entry['path']
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != entry['sha256']:
        raise ValueError(f"report changed: {entry['path']}")
    report = json.loads(raw)
    if report.get('passed') is not True or not report.get('source_sha256'):
        raise ValueError(f"report has no passing source-backed result: {entry['path']}")
    revision = None if current else entry.get('source_revision')
    if revision is not None and not re.fullmatch('[0-9a-f]{40}', revision):
        raise ValueError('source_revision must be a full immutable Git commit SHA')
    for source, expected in report['source_sha256'].items():
        # Git checkouts on Windows may use CRLF; execution bundles use LF.
        if revision is None:
            content = (root / source).read_bytes().replace(b'\r\n', b'\n')
        else:
            result = subprocess.run(['git', 'show', f'{revision}:{source}'], cwd=root,
                                    capture_output=True)
            if result.returncode:
                raise ValueError(f'cannot read historical source {revision}:{source}; '
                                 'use a Git checkout with this commit fetched')
            content = result.stdout
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(f"source differs from recorded run: {source}")
    return len(report['source_sha256'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, default=Path('docs/results/v0.1/evidence.json'))
    parser.add_argument('--current', action='store_true',
                        help='compare with working files instead of recorded historical revisions')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    index = json.loads((root / args.index).read_text(encoding='utf-8'))
    for entry in index['reports']:
        count = verify(root, entry, current=args.current)
        target = entry.get('source_revision') if not args.current else None
        print(f"PASS: {entry['path']} ({count} source hashes match {target or 'working files'})")
    print('No hardware test was rerun. Historical matches certify only their recorded source.')


if __name__ == '__main__':
    main()
