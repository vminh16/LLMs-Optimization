from argparse import Namespace
import unittest
from unittest.mock import patch

from scripts import summarize_experiment


def group(tbt: float, *, ready: bool = True) -> dict:
    return {
        "ready_for_comparison": ready,
        "blocking_issues": [] if ready else ["not enough runs"],
        "identity": {"measurement_version": "h0.1", "trace_sha256": "trace"},
        "metrics": {
            "tbt_median_ms": {"median": tbt},
            "ttft_p95_ms": {"median": 1000.0},
            "makespan_ms": {"median": 10000.0},
        },
    }


class SummarizeExperimentScriptTest(unittest.TestCase):
    def args(self) -> Namespace:
        return Namespace(
            baseline_root="baseline",
            experiment_root="experiment",
            min_runs=1,
            expected_total_count=120,
            output=None,
        )

    @patch.object(summarize_experiment, "summarize_candidate_groups")
    @patch.object(summarize_experiment, "summarize_run_group")
    def test_renderer_compares_with_mm_cache_off_control(self, baseline_summary, candidate_summary):
        baseline_summary.return_value = group(100.0)
        candidate_summary.return_value = {
            "groups": {
                "mm-cache-off": group(80.0),
                "renderer-2": group(72.0),
            }
        }

        report = summarize_experiment.build_report(self.args())

        comparison = report["comparisons"]["renderer-2"]
        self.assertEqual(comparison["comparison_control"], "mm-cache-off")
        self.assertAlmostEqual(comparison["changes_pct"]["tbt_median_ms"], -10.0)

    @patch.object(summarize_experiment, "summarize_candidate_groups")
    @patch.object(summarize_experiment, "summarize_run_group")
    def test_renderer_is_blocked_when_control_is_not_ready(self, baseline_summary, candidate_summary):
        baseline_summary.return_value = group(100.0)
        candidate_summary.return_value = {
            "groups": {
                "mm-cache-off": group(80.0, ready=False),
                "renderer-2": group(72.0),
            }
        }

        report = summarize_experiment.build_report(self.args())

        comparison = report["comparisons"]["renderer-2"]
        self.assertEqual(comparison["decision"], "blocked")
        self.assertIn("control mm-cache-off", comparison["blocking_issues"][0])


if __name__ == "__main__":
    unittest.main()
