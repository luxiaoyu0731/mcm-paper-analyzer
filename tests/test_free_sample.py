import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class FreeSampleTest(unittest.TestCase):
    def test_report_citations_and_existing_destination_protection(self):
        script = Path(__file__).resolve().parents[1] / 'examples/try_retrieval.py'
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'new-sample'
            first = subprocess.run([sys.executable,str(script),'--destination',str(target)],capture_output=True,text=True)
            self.assertEqual(first.returncode,0,first.stderr)
            report = (target/'sample-report.md').read_text(encoding='utf-8')
            rows = json.loads((target/'retrieval.json').read_text(encoding='utf-8'))
            self.assertEqual(len(rows),2)
            for row in rows:self.assertIn(row['chunk_id'],report)
            self.assertIn('manually authored',report)
            before=(target/'sample-report.md').read_bytes()
            second=subprocess.run([sys.executable,str(script),'--destination',str(target)],capture_output=True)
            self.assertNotEqual(second.returncode,0)
            self.assertEqual((target/'sample-report.md').read_bytes(),before)
