import asyncio
import unittest

from inference_opt.benchmark.runner import BenchmarkConfig, run_benchmark
from inference_opt.trace.loader import TraceRecord


async def fake_sender(body):
    if body["messages"][0]["content"] == "fail":
        raise RuntimeError("boom")
    return {
        "ttft_ms": 100.0,
        "tpot_ms": 20.0,
        "output_tokens": 2,
        "text": "ok",
    }


class BenchmarkRunnerTest(unittest.TestCase):
    def test_run_benchmark_scores_success_and_error_records(self):
        records = [
            TraceRecord(
                request_id=1,
                timestamp_ms=0,
                workload_type="conversation",
                body={"messages": [{"role": "user", "content": "ok"}]},
            ),
            TraceRecord(
                request_id=2,
                timestamp_ms=0,
                workload_type="conversation",
                body={"messages": [{"role": "user", "content": "fail"}]},
            ),
        ]

        result = asyncio.run(
            run_benchmark(
                records,
                BenchmarkConfig(base_url="http://localhost:8000/v1", request_timeout_s=1.0),
                sender=fake_sender,
                respect_timestamps=False,
            )
        )

        self.assertEqual(result.summary["request_count"], 2)
        self.assertEqual(result.summary["success_count"], 1)
        self.assertEqual(result.summary["error_count"], 1)
        self.assertEqual(result.summary["effective_request_score"], 0.5)
        self.assertEqual(result.requests[0]["score"], 1.0)
        self.assertEqual(result.requests[0]["text"], "ok")
        self.assertEqual(result.requests[1]["score"], 0.0)
        self.assertIn("boom", result.requests[1]["error"])


if __name__ == "__main__":
    unittest.main()
