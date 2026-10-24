"""A passing historical report must not silently certify different source."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import shutil
import subprocess

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

    @unittest.skipUnless(shutil.which('git'), 'historical checks require Git')
    def test_immutable_history_does_not_certify_changed_working_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = self.fixture(root)
            def git(*args):
                return subprocess.check_output(['git', *args], cwd=root, stderr=subprocess.DEVNULL)
            git('init', '-q')
            git('add', 'source.py')
            git('-c', 'user.name=Test', '-c', 'user.email=test@example.org', 'commit', '-qm', 'reference')
            entry['source_revision'] = git('rev-parse', 'HEAD').decode().strip()
            (root/'source.py').write_text('print(2)\n')
            self.assertEqual(verify(root, entry), 1)
            with self.assertRaisesRegex(ValueError, 'source differs'):
                verify(root, entry, current=True)
            git('add', 'source.py')
            git('-c', 'user.name=Test', '-c', 'user.email=test@example.org', 'commit', '-qm', 'changed')
            entry['source_revision'] = git('rev-parse', 'HEAD').decode().strip()
            with self.assertRaisesRegex(ValueError, 'source differs'):
                verify(root, entry)
            entry['source_revision'] = '0'*40
            with self.assertRaisesRegex(ValueError, 'cannot read historical source'):
                verify(root, entry)

    def test_moving_revision_names_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = self.fixture(root)
            entry['source_revision'] = 'main'
            with self.assertRaisesRegex(ValueError, 'immutable Git commit'):
                verify(root, entry)


if __name__ == '__main__':
    unittest.main()
