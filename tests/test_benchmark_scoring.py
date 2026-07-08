import unittest

from inference_opt.benchmark.scoring import (
    ScoreConfig,
    RequestMeasurement,
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


if __name__ == "__main__":
    unittest.main()
