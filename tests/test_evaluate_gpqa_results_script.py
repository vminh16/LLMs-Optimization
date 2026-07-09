import unittest

from scripts.evaluate_gpqa_results import build_report
from inference_opt.eval.gpqa import LOCAL_BASELINE_ACCURACY


class EvaluateGPQAResultsScriptTest(unittest.TestCase):
    def test_build_report_uses_measured_local_baseline(self):
        answer_key = {request_id: "A" for request_id in range(120)}
        rows = [
            {
                "request_id": request_id,
                "text": "A" if request_id < 43 else "B",
            }
            for request_id in range(120)
        ]

        report = build_report(answer_key, rows)

        self.assertEqual(LOCAL_BASELINE_ACCURACY, 43 / 120)
        self.assertEqual(report["accuracy"], 43 / 120)
        self.assertEqual(report["baseline_accuracy"], 43 / 120)
        self.assertEqual(report["accuracy_delta"], 0.0)
        self.assertEqual(report["accuracy_multiplier"], 1.0)

    def test_build_report_applies_local_penalty_boundaries(self):
        answer_key = {request_id: "A" for request_id in range(120)}

        def rows_with_correct_count(correct_count: int) -> list[dict]:
            return [
                {
                    "request_id": request_id,
                    "text": "A" if request_id < correct_count else "B",
                }
                for request_id in range(120)
            ]

        full_credit = build_report(answer_key, rows_with_correct_count(31))
        penalized = build_report(answer_key, rows_with_correct_count(30))
        zero_credit = build_report(answer_key, rows_with_correct_count(23))

        self.assertEqual(full_credit["accuracy_multiplier"], 1.0)
        self.assertGreater(penalized["accuracy_multiplier"], 0.0)
        self.assertLess(penalized["accuracy_multiplier"], 1.0)
        self.assertEqual(zero_credit["accuracy_multiplier"], 0.0)


if __name__ == "__main__":
    unittest.main()
