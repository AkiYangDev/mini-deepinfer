"""A passing historical report must not silently certify different source."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.check_evidence import verify


class EvidenceTests(unittest.TestCase):
    def fixture(self, root, passed=True):
        source = b'print(1)\n'
        (root / 'source.py').write_bytes(source)
        report = json.dumps(dict(passed=passed, source_sha256={
            'source.py': hashlib.sha256(source).hexdigest()})).encode()
        (root / 'report.json').write_bytes(report)
        return dict(path='report.json', sha256=hashlib.sha256(report).hexdigest())

    def test_matching_source_accepts_checkout_line_endings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = self.fixture(root)
            self.assertEqual(verify(root, entry), 1)
            (root / 'source.py').write_bytes(b'print(1)\r\n')
            self.assertEqual(verify(root, entry), 1)

    def test_changed_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = self.fixture(root)
            (root / 'source.py').write_text('print(2)\n')
            with self.assertRaisesRegex(ValueError, 'source differs'):
                verify(root, entry)

    def test_changed_or_failed_report_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = self.fixture(root, passed=False)
            with self.assertRaisesRegex(ValueError, 'no passing'):
                verify(root, entry)
            (root / 'report.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'report changed'):
                verify(root, entry)


if __name__ == '__main__':
    unittest.main()
