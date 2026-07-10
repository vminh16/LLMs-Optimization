import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from inference_opt.benchmark import report


def write_summary(root: Path, run_id: str, **overrides):
    summary = {
        "total_count": 120,
        "failed_count": 0,
        "ttft_p50_ms": 1000.0,
        "ttft_p95_ms": 3000.0,
        "tbt_median_ms": 40.0,
        "ers": 10.0,
        "final_score": 10.0,
    }
    summary.update(overrides)
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


class BenchmarkReportTest(unittest.TestCase):
    def test_summarize_run_group_reports_median_mad_and_ready_state(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_summary(root, "baseline-cold-01", ttft_p50_ms=1000.0, tbt_median_ms=40.0, ers=10.0)
            write_summary(root, "baseline-cold-02", ttft_p50_ms=1100.0, tbt_median_ms=42.0, ers=11.0)
            write_summary(root, "baseline-cold-03", ttft_p50_ms=900.0, tbt_median_ms=38.0, ers=9.0)

            result = report.summarize_run_group(root, min_runs=3, expected_total_count=120)

        self.assertTrue(result["ready_for_comparison"])
        self.assertEqual(result["run_count"], 3)
        self.assertEqual(result["run_ids"], ["baseline-cold-01", "baseline-cold-02", "baseline-cold-03"])
        self.assertEqual(result["blocking_issues"], [])
        self.assertEqual(result["metrics"]["ttft_p50_ms"]["median"], 1000.0)
        self.assertEqual(result["metrics"]["ttft_p50_ms"]["mad"], 100.0)
        self.assertEqual(result["metrics"]["tbt_median_ms"]["median"], 40.0)
        self.assertEqual(result["metrics"]["ers"]["median"], 10.0)

    def test_summarize_run_group_blocks_when_runs_are_incomplete(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_summary(root, "baseline-cold-01", failed_count=0)
            write_summary(root, "baseline-cold-02", failed_count=1)

            result = report.summarize_run_group(root, min_runs=3, expected_total_count=120)

        self.assertFalse(result["ready_for_comparison"])
        self.assertIn("expected at least 3 runs, found 2", result["blocking_issues"])
        self.assertIn("baseline-cold-02 has failed_count=1", result["blocking_issues"])

    def test_summarize_candidate_groups_keeps_candidates_separate(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_summary(root, "kv-fp8-seqs-32-01", ers=8.0)
            write_summary(root, "kv-fp8-seqs-32-02", ers=10.0)
            write_summary(root, "kv-fp8-seqs-64-01", ers=20.0)
            write_summary(root, "kv-fp8-seqs-64-02", ers=22.0)

            result = report.summarize_candidate_groups(root, min_runs=2, expected_total_count=120)

        self.assertEqual(sorted(result["groups"]), ["kv-fp8-seqs-32", "kv-fp8-seqs-64"])
        self.assertEqual(result["groups"]["kv-fp8-seqs-32"]["metrics"]["ers"]["median"], 9.0)
        self.assertEqual(result["groups"]["kv-fp8-seqs-64"]["metrics"]["ers"]["median"], 21.0)


if __name__ == "__main__":
    unittest.main()
