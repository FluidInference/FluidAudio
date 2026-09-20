"""Scorer tests and corruption checks against a real benchmark report."""

from copy import deepcopy
import json
import os
from pathlib import Path
import unittest

from verify_localvqe_benchmark import edits, merge, verify


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


def split_into_shards(report, count):
    """Re-slice a full report the way enhance-benchmark --shard i/n selects examples."""
    selected = report["dataset"]["selected_fileids"]
    size = -(-len(selected) // count)
    shards = []
    for index in range(count):
        ids = selected[index * size:(index + 1) * size]
        shard = deepcopy(report)
        shard["shard"] = {"index": index, "count": count}
        shard["dataset"]["selected_fileids"] = ids
        shard["dataset"]["audio_files_sha256"] = {
            name: digest for name, digest in report["dataset"]["audio_files_sha256"].items()
            if name.split("_")[1] in ids}
        shard["files"] = [row for row in report["files"] if row["fileid"] in ids]
        shard["excluded_empty_reference_fileids"] = [i for i in report["excluded_empty_reference_fileids"] if i in ids]
        for condition, summary in shard["summary"].items():
            rows = shard["files"]
            for field, key in (("files", None), ("reference_words", "ref_words"), ("far_end_words", "far_words"),
                               ("hits", f"{condition}_hits"), ("errors", f"{condition}_errors"),
                               ("leaked_words", f"{condition}_leaked"), ("audio_seconds", "audio_seconds"),
                               ("enhancement_seconds", f"{condition}_enhancement_seconds")):
                summary[field] = len(rows) if key is None else sum(row[key] for row in rows)
        shards.append(shard)
    return shards


class MergeTests(unittest.TestCase):
    def shard(self, index, count, **overrides):
        base = {
            "schema_version": 2, "protocol": "localvqe-asr-v2", "configuration": {"asr": "x"}, "chunk": "256ms",
            "conditions": ["unprocessed"], "model_files_sha256": {"m": "0" * 64},
            "dataset": {"path": "p", "repository": "r", "revision": "v", "archive_sha256": "a", "metadata_sha256": "b",
                        "selection_order": "numeric fileid", "selected_fileids": [str(index)],
                        "audio_files_sha256": {f"fileid_{index}_mic.wav": "0" * 64}},
            "files": [{"fileid": str(index)}], "excluded_empty_reference_fileids": [],
            "started_at": f"2026-01-01T00:0{index}:00Z", "completed_at": f"2026-01-01T00:0{index}:30Z",
            "environment": {"source_revision": "abc"}, "shard": {"index": index, "count": count},
            "summary": {"unprocessed": {"files": 1, "reference_words": 10, "hits": 5, "errors": 6, "far_end_words": 4,
                                        "leaked_words": 1, "audio_seconds": 8.0, "enhancement_seconds": 0.0}},
        }
        base.update(overrides)
        return base

    def test_merges_in_index_order_regardless_of_input_order(self):
        merged = merge([self.shard(1, 2), self.shard(0, 2)])
        self.assertEqual(merged["dataset"]["selected_fileids"], ["0", "1"])
        self.assertEqual([row["fileid"] for row in merged["files"]], ["0", "1"])
        self.assertEqual(merged["summary"]["unprocessed"]["hits"], 10)
        self.assertAlmostEqual(merged["summary"]["unprocessed"]["recall"], 0.5)
        self.assertEqual(merged["started_at"], "2026-01-01T00:00:00Z")
        self.assertEqual(merged["completed_at"], "2026-01-01T00:01:30Z")
        self.assertEqual([s["index"] for s in merged["shards"]], [0, 1])
        self.assertIsNone(merged["shard"])

    def test_missing_duplicate_or_mismatched_shards_fail(self):
        with self.assertRaisesRegex(ValueError, "Expected shards"):
            merge([self.shard(0, 3), self.shard(1, 3)])
        with self.assertRaisesRegex(ValueError, "Expected shards"):
            merge([self.shard(0, 2), self.shard(0, 2)])
        with self.assertRaisesRegex(ValueError, "shard count"):
            merge([self.shard(0, 2), self.shard(1, 3)])
        with self.assertRaisesRegex(ValueError, "model_files_sha256"):
            merge([self.shard(0, 2), self.shard(1, 2, model_files_sha256={"m": "1" * 64})])
        with self.assertRaisesRegex(ValueError, "source revisions"):
            merge([self.shard(0, 2), self.shard(1, 2, environment={"source_revision": "other"})])
        with self.assertRaisesRegex(ValueError, "not a shard"):
            merge([self.shard(0, 1, shard=None)])

    def test_single_shard_report_is_rejected_by_verify(self):
        with self.assertRaisesRegex(ValueError, "Shard 0/2"):
            verify(self.shard(0, 2), 1)


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

    def test_shards_merge_back_to_the_full_report(self):
        merged = merge(split_into_shards(self.report, 5))
        self.assertEqual(merged["dataset"]["selected_fileids"], self.report["dataset"]["selected_fileids"])
        self.assertEqual(merged["files"], self.report["files"])
        self.assertEqual(merged["excluded_empty_reference_fileids"], self.report["excluded_empty_reference_fileids"])
        for condition, summary in self.report["summary"].items():
            for field, value in summary.items():
                self.assertAlmostEqual(merged["summary"][condition][field], value, places=9, msg=f"{condition}.{field}")
        verify(merged, self.count)


if __name__ == "__main__":
    unittest.main()
