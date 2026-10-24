"""Check preserved NPU results against this checkout, without using a device."""

import hashlib
import json
from pathlib import Path


def verify(root, entry):
    path = root / entry['path']
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != entry['sha256']:
        raise ValueError(f"report changed: {entry['path']}")
    report = json.loads(raw)
    if report.get('passed') is not True or not report.get('source_sha256'):
        raise ValueError(f"report has no passing source-backed result: {entry['path']}")
    for source, expected in report['source_sha256'].items():
        # Git checkouts on Windows may use CRLF; execution bundles use LF.
        content = (root / source).read_bytes().replace(b'\r\n', b'\n')
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(f"source differs from hardware run: {source}")
    return len(report['source_sha256'])


def main():
    root = Path(__file__).resolve().parent.parent
    index = json.loads((root / 'docs/results/v0.1/evidence.json').read_text(encoding='utf-8'))
    for entry in index['reports']:
        count = verify(root, entry)
        print(f"PASS: {entry['path']} ({count} source hashes match)")
    print('Preserved evidence matches this checkout; no hardware test was rerun.')


if __name__ == '__main__':
    main()
