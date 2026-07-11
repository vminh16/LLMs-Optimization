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
        "erc": 0.5,
        "makespan_ms": 10000.0,
        "dispatch_lag_p95_ms": 5.0,
        "prompt_tokens_p50": 1000,
        "prompt_tokens_p95": 2000,
        "measurement_version": "h0.1",
        "trace_sha256": "same-trace",
        "ers": 10.0,
        "final_score": 10.0,
    }
    summary.update(overrides)
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


def write_manifest(root: Path, run_id: str, *, candidate: str, status: str = "completed"):
    (root / run_id / "experiment.json").write_text(
        json.dumps({"run_id": run_id, "candidate": candidate, "status": status}),
        encoding="utf-8",
    )


class BenchmarkReportTest(unittest.TestCase):
    def test_compare_advances_meaningful_tbt_gain_without_makespan_regression(self):
        baseline = {"tbt_median_ms": 100.0, "ttft_p95_ms": 1000.0, "makespan_ms": 10000.0}
        candidate = {"tbt_median_ms": 98.0, "ttft_p95_ms": 1000.0, "makespan_ms": 10100.0}

        result = report.compare_candidate_to_baseline(baseline, candidate)

        self.assertEqual(result["decision"], "advance")
        self.assertAlmostEqual(result["changes_pct"]["tbt_median_ms"], -2.0)

    def test_compare_rejects_material_tbt_regression(self):
        baseline = {"tbt_median_ms": 100.0, "ttft_p95_ms": 1000.0, "makespan_ms": 10000.0}
        candidate = {"tbt_median_ms": 102.0, "ttft_p95_ms": 950.0, "makespan_ms": 9900.0}

        result = report.compare_candidate_to_baseline(baseline, candidate)

        self.assertEqual(result["decision"], "reject")

    def test_compare_marks_small_changes_inconclusive(self):
        baseline = {"tbt_median_ms": 100.0, "ttft_p95_ms": 1000.0, "makespan_ms": 10000.0}
        candidate = {"tbt_median_ms": 99.5, "ttft_p95_ms": 980.0, "makespan_ms": 10050.0}

        result = report.compare_candidate_to_baseline(baseline, candidate)

        self.assertEqual(result["decision"], "inconclusive")

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
        self.assertEqual(result["metrics"]["erc"]["median"], 0.5)
        self.assertEqual(result["metrics"]["makespan_ms"]["median"], 10000.0)
        self.assertEqual(result["metrics"]["dispatch_lag_p95_ms"]["median"], 5.0)
        self.assertEqual(result["metrics"]["ers"]["median"], 10.0)
        self.assertEqual(result["identity"], {"measurement_version": "h0.1", "trace_sha256": "same-trace"})

    def test_summarize_run_group_blocks_when_runs_are_incomplete(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_summary(root, "baseline-cold-01", failed_count=0)
            write_summary(root, "baseline-cold-02", failed_count=1)

            result = report.summarize_run_group(root, min_runs=3, expected_total_count=120)

        self.assertFalse(result["ready_for_comparison"])
        self.assertIn("expected at least 3 runs, found 2", result["blocking_issues"])
        self.assertIn("baseline-cold-02 has failed_count=1", result["blocking_issues"])

    def test_summarize_run_group_blocks_mixed_trace_hashes(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_summary(root, "baseline-cold-01", trace_sha256="trace-a")
            write_summary(root, "baseline-cold-02", trace_sha256="trace-b")

            result = report.summarize_run_group(root, min_runs=2, expected_total_count=120)

        self.assertFalse(result["ready_for_comparison"])
        self.assertIn("mixed trace_sha256 values", result["blocking_issues"][0])

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

    def test_candidate_group_blocks_missing_or_mismatched_manifest(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_summary(root, "renderer-2-01")
            write_summary(root, "language-only-01")
            write_manifest(root, "language-only-01", candidate="wrong-candidate")

            result = report.summarize_candidate_groups(
                root,
                min_runs=1,
                expected_total_count=120,
                require_manifests=True,
            )

        self.assertFalse(result["groups"]["renderer-2"]["ready_for_comparison"])
        self.assertIn("missing experiment.json", result["groups"]["renderer-2"]["blocking_issues"][0])
        self.assertFalse(result["groups"]["language-only"]["ready_for_comparison"])
        self.assertIn("candidate mismatch", result["groups"]["language-only"]["blocking_issues"][0])

    def test_candidate_group_blocks_invalid_manifest_json(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_summary(root, "renderer-2-01")
            (root / "renderer-2-01" / "experiment.json").write_text("not-json", encoding="utf-8")

            result = report.summarize_candidate_groups(
                root,
                min_runs=1,
                expected_total_count=120,
                require_manifests=True,
            )

        self.assertFalse(result["groups"]["renderer-2"]["ready_for_comparison"])
        self.assertIn("invalid experiment.json", result["groups"]["renderer-2"]["blocking_issues"][0])

    def test_run_group_blocks_missing_comparison_metric(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_summary(root, "renderer-2-01", makespan_ms=None)

            result = report.summarize_run_group(root, min_runs=1, expected_total_count=120)

        self.assertFalse(result["ready_for_comparison"])
        self.assertIn("renderer-2-01 is missing makespan_ms", result["blocking_issues"])


if __name__ == "__main__":
    unittest.main()
