import json
import tempfile
import unittest
from pathlib import Path

from inference_opt.eval.gpqa import GPQAQuestion
from inference_opt.trace.gpqa_trace import (
    build_gpqa_trace_records,
    write_gpqa_trace_bundle,
)
from inference_opt.trace.loader import load_trace


class GPQATraceBuilderTest(unittest.TestCase):
    def test_build_gpqa_trace_records_uses_schedule_and_openai_body(self):
        questions = [
            GPQAQuestion(
                request_id=0,
                question="Question zero?",
                choices={"A": "alpha", "B": "beta", "C": "gamma", "D": "delta"},
                correct_label="B",
            ),
            GPQAQuestion(
                request_id=1,
                question="Question one?",
                choices={"A": "red", "B": "green", "C": "blue", "D": "black"},
                correct_label="C",
            ),
        ]
        schedule = [(10, 0), (11, 250)]

        records = build_gpqa_trace_records(
            questions,
            schedule=schedule,
            model="Qwen3.5-2B",
            max_tokens=8,
        )

        self.assertEqual([record.request_id for record in records], [10, 11])
        self.assertEqual([record.timestamp_ms for record in records], [0, 250])
        self.assertEqual(records[0].workload_type, "gpqa_diamond")
        self.assertEqual(records[0].body["model"], "Qwen3.5-2B")
        self.assertEqual(records[0].body["max_tokens"], 8)
        self.assertEqual(records[0].body["temperature"], 0)
        self.assertEqual(records[0].body["messages"][0]["role"], "system")
        self.assertIn("Question zero?", records[0].body["messages"][1]["content"])
        self.assertIn("A. alpha", records[0].body["messages"][1]["content"])

    def test_write_gpqa_trace_bundle_writes_trace_and_answer_key(self):
        questions = [
            GPQAQuestion(
                request_id=0,
                question="Question zero?",
                choices={"A": "alpha", "B": "beta", "C": "gamma", "D": "delta"},
                correct_label="B",
            )
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            trace_path = Path(tmpdir) / "gpqa.jsonl"
            answer_key_path = Path(tmpdir) / "gpqa.answers.json"

            write_gpqa_trace_bundle(
                questions,
                trace_path=trace_path,
                answer_key_path=answer_key_path,
                schedule=[(5, 100)],
                model="Qwen3.5-2B",
            )

            records = load_trace(trace_path)
            answer_key = json.loads(answer_key_path.read_text(encoding="utf-8"))

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].request_id, 5)
        self.assertEqual(answer_key["answers"], [{"request_id": 5, "correct_label": "B"}])


if __name__ == "__main__":
    unittest.main()
