"""Scorer tests and corruption checks against a real benchmark report."""

from copy import deepcopy
import json
import os
from pathlib import Path
import unittest

from verify_localvqe_benchmark import edits, verify


class EditTests(unittest.TestCase):
    def test_empty_hypothesis_is_deletions(self):
        self.assertEqual(edits([], ["one", "two"]), (0, 2, 0))

    def test_empty_reference_is_insertions(self):
        self.assertEqual(edits(["one", "two"], []), (0, 0, 2))

    def test_mixed_edits(self):
        self.assertEqual(edits("a x c d e".split(), "a b c d".split()), (1, 0, 1))
        self.assertEqual(edits("a c".split(), "a b c d".split()), (0, 2, 0))

    def test_substitution_wins_ties(self):
        self.assertEqual(edits(["a", "b"], ["b", "a"]), (2, 0, 0))


class RealReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = os.environ.get("LOCALVQE_BENCHMARK_REPORT")
        if not path:
            raise unittest.SkipTest("Set LOCALVQE_BENCHMARK_REPORT to a real enhance-benchmark JSON report")
        cls.original = json.loads(Path(path).read_text())
        cls.count = len(cls.original["dataset"]["selected_fileids"])

    def setUp(self):
        self.report = deepcopy(self.original)

    def test_real_report_passes(self):
        verify(self.report, self.count)

    def test_missing_row_fails(self):
        self.report["files"].pop()
        with self.assertRaises(ValueError):
            verify(self.report, self.count)

    def test_wrong_summary_fails(self):
        self.report["summary"]["localvqe-v1.3"]["recall"] += 0.1
        with self.assertRaises(ValueError):
            verify(self.report, self.count)

    def test_wrong_edit_count_fails(self):
        self.report["files"][0]["localvqe-v1.3_deletions"] += 1
        with self.assertRaises(ValueError):
            verify(self.report, self.count)

    def test_modified_audio_fails(self):
        name = next(iter(self.report["dataset"]["audio_files_sha256"]))
        self.report["dataset"]["audio_files_sha256"][name] = "0" * 64
        with self.assertRaises(ValueError):
            verify(self.report, self.count)

    def test_excluded_and_scored_overlap_fails(self):
        self.report["excluded_empty_reference_fileids"].append(self.report["files"][0]["fileid"])
        with self.assertRaises(ValueError):
            verify(self.report, self.count)

    def test_missing_model_fingerprints_fail(self):
        self.report["model_files_sha256"] = {}
        with self.assertRaises(ValueError):
            verify(self.report, self.count)

    def test_missing_condition_fails(self):
        self.report["conditions"].pop()
        with self.assertRaises(ValueError):
            verify(self.report, self.count)


if __name__ == "__main__":
    unittest.main()
