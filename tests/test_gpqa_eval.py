import unittest

from inference_opt.eval.gpqa import (
    answer_letter_from_text,
    normalize_gpqa_row,
    score_gpqa_results,
)


class GPQAEvalTest(unittest.TestCase):
    def test_normalize_gpqa_row_builds_deterministic_choices(self):
        row = {
            "Question": "Which particle has no electric charge?",
            "Correct Answer": "Neutron",
            "Incorrect Answer 1": "Proton",
            "Incorrect Answer 2": "Electron",
            "Incorrect Answer 3": "Muon",
        }

        first = normalize_gpqa_row(row, request_id=3, shuffle_seed=11)
        second = normalize_gpqa_row(row, request_id=3, shuffle_seed=11)

        self.assertEqual(first, second)
        self.assertEqual(first.request_id, 3)
        self.assertEqual(first.question, row["Question"])
        self.assertEqual(len(first.choices), 4)
        self.assertIn(first.correct_label, {"A", "B", "C", "D"})
        self.assertEqual(first.choices[first.correct_label], "Neutron")

    def test_normalize_gpqa_row_rejects_incomplete_rows(self):
        row = {
            "Question": "Incomplete?",
            "Correct Answer": "Yes",
            "Incorrect Answer 1": "No",
        }

        with self.assertRaises(ValueError):
            normalize_gpqa_row(row, request_id=1, shuffle_seed=1)

    def test_answer_letter_from_text_extracts_common_formats(self):
        self.assertEqual(answer_letter_from_text("B"), "B")
        self.assertEqual(answer_letter_from_text("Answer: c"), "C")
        self.assertEqual(answer_letter_from_text("The answer is (D)."), "D")
        self.assertIsNone(answer_letter_from_text("I cannot determine this."))

    def test_score_gpqa_results_counts_missing_and_wrong_answers(self):
        answer_key = {
            0: "A",
            1: "C",
            2: "D",
        }
        rows = [
            {"request_id": 0, "text": "Answer: A"},
            {"request_id": 1, "text": "B"},
        ]

        report = score_gpqa_results(answer_key, rows)

        self.assertEqual(report["question_count"], 3)
        self.assertEqual(report["answered_count"], 2)
        self.assertEqual(report["correct_count"], 1)
        self.assertEqual(report["missing_count"], 1)
        self.assertAlmostEqual(report["accuracy"], 1 / 3)


if __name__ == "__main__":
    unittest.main()
