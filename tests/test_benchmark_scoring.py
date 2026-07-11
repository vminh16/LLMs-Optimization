import unittest

from inference_opt.benchmark.scoring import (
    ScoreConfig,
    RequestMeasurement,
    accuracy_multiplier,
    final_score,
    request_score,
    summarize_scores,
)


class BenchmarkScoringTest(unittest.TestCase):
    def test_request_score_returns_zero_for_error_or_no_tokens(self):
        config = ScoreConfig()

        self.assertEqual(
            request_score(RequestMeasurement(error="timeout", output_tokens=3), config),
            0.0,
        )
        self.assertEqual(
            request_score(RequestMeasurement(ttft_ms=100.0, tpot_ms=20.0, output_tokens=0), config),
            0.0,
        )

    def test_request_score_matches_floor_and_ceiling_behavior(self):
        config = ScoreConfig()

        perfect = request_score(
            RequestMeasurement(ttft_ms=100.0, tpot_ms=20.0, output_tokens=10),
            config,
        )
        failed = request_score(
            RequestMeasurement(ttft_ms=1500.0, tpot_ms=45.0, output_tokens=10),
            config,
        )

        self.assertEqual(perfect, 1.0)
        self.assertEqual(failed, 0.0)

    def test_request_score_squares_ttft_and_tpot_terms(self):
        config = ScoreConfig()

        score = request_score(
            RequestMeasurement(ttft_ms=800.0, tpot_ms=32.5, output_tokens=8),
            config,
        )

        self.assertAlmostEqual(score, 0.25)

    def test_summarize_scores_reports_effective_request_score(self):
        config = ScoreConfig()
        measurements = [
            RequestMeasurement(ttft_ms=100.0, tpot_ms=20.0, output_tokens=2),
            RequestMeasurement(error="http 500", output_tokens=0),
        ]

        summary = summarize_scores(measurements, config)

        self.assertEqual(summary["request_count"], 2)
        self.assertEqual(summary["success_count"], 1)
        self.assertEqual(summary["error_count"], 1)
        self.assertEqual(summary["effective_request_score"], 0.5)

    def test_summarize_scores_reports_btc_like_latency_fields(self):
        config = ScoreConfig()
        measurements = [
            RequestMeasurement(ttft_ms=100.0, tpot_ms=20.0, output_tokens=2),
            RequestMeasurement(ttft_ms=200.0, tpot_ms=30.0, output_tokens=2),
            RequestMeasurement(ttft_ms=2000.0, tpot_ms=50.0, output_tokens=2),
            RequestMeasurement(error="timeout", output_tokens=0),
        ]

        summary = summarize_scores(measurements, config)

        self.assertEqual(summary["total_count"], 4)
        self.assertEqual(summary["failed_count"], 1)
        self.assertEqual(summary["passed_slo"], 2)
        self.assertEqual(summary["ttft_p50_ms"], 200.0)
        self.assertEqual(summary["ttft_p95_ms"], 2000.0)
        self.assertEqual(summary["tbt_median_ms"], 30.0)
        self.assertAlmostEqual(summary["ers"], 100.0 * summary["effective_request_score"])
        self.assertAlmostEqual(summary["final_score"], summary["ers"])

    def test_summarize_scores_reports_h01_observability_fields(self):
        config = ScoreConfig()
        measurements = [
            RequestMeasurement(
                ttft_ms=100.0,
                tpot_ms=20.0,
                prompt_tokens=100,
                output_tokens=2,
                dispatch_lag_ms=1.0,
            ),
            RequestMeasurement(
                prompt_tokens=200,
                output_tokens=0,
                error="timeout",
                dispatch_lag_ms=7.0,
            ),
        ]

        summary = summarize_scores(measurements, config)

        self.assertEqual(summary["erc"], 0.5)
        self.assertEqual(summary["prompt_tokens_p50"], 100)
        self.assertEqual(summary["prompt_tokens_p95"], 200)
        self.assertEqual(summary["dispatch_lag_p95_ms"], 7.0)

    def test_accuracy_multiplier_matches_project_gate(self):
        self.assertEqual(accuracy_multiplier(accuracy=0.30), 1.0)
        self.assertAlmostEqual(accuracy_multiplier(accuracy=0.27), 0.5)
        self.assertEqual(accuracy_multiplier(accuracy=0.24), 0.0)

    def test_final_score_combines_ers_and_accuracy_gate(self):
        self.assertEqual(final_score(effective_request_score=0.5, accuracy=0.30), 50.0)
        self.assertEqual(final_score(effective_request_score=0.5, accuracy=0.24), 0.0)


if __name__ == "__main__":
    unittest.main()
