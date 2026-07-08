from pathlib import Path
import tempfile
import unittest

from inference_opt.trace.loader import load_trace


class TraceLoaderTest(unittest.TestCase):
    def test_load_trace_preserves_request_metadata_and_body(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_path = Path(tmpdir) / "trace.jsonl"
            trace_path.write_text(
                '{"request_id": 7, "timestamp_ms": 12, "workload_type": "conversation", '
                '"body": {"model": "Qwen3.5-2B", "messages": [{"role": "user", "content": "hi"}]}}\n',
                encoding="utf-8",
            )

            records = load_trace(trace_path)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].request_id, 7)
        self.assertEqual(records[0].timestamp_ms, 12)
        self.assertEqual(records[0].workload_type, "conversation")
        self.assertEqual(records[0].body["model"], "Qwen3.5-2B")


if __name__ == "__main__":
    unittest.main()
