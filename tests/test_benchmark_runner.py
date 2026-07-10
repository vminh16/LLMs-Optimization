import asyncio
import json
import unittest
from unittest.mock import patch

import httpx

from inference_opt.benchmark.runner import BenchmarkConfig, run_benchmark, send_openai_streaming_request
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
    def test_streaming_request_uses_usage_tokens_for_tpot(self):
        seen_bodies = []

        async def handler(request):
            seen_bodies.append(request.read())
            lines = [
                b'data: {"choices":[{"delta":{"content":"hello "}}]}\n\n',
                b'data: {"choices":[{"delta":{"content":"world"}}]}\n\n',
                b'data: {"choices":[],"usage":{"completion_tokens":5}}\n\n',
                b"data: [DONE]\n\n",
            ]
            return httpx.Response(200, content=b"".join(lines))

        transport = httpx.MockTransport(handler)
        clock = iter([10.0, 10.2, 10.4, 11.2])

        async def run():
            async with httpx.AsyncClient(transport=transport) as client:
                return await send_openai_streaming_request(
                    {"messages": [{"role": "user", "content": "hi"}]},
                    base_url="http://test/v1",
                    timeout_s=1.0,
                    client=client,
                )

        with patch("inference_opt.benchmark.runner.perf_counter", side_effect=lambda: next(clock)):
            result = asyncio.run(run())

        self.assertEqual(result["output_tokens"], 5)
        self.assertAlmostEqual(result["ttft_ms"], 200.0)
        self.assertAlmostEqual(result["tpot_ms"], 250.0)
        self.assertEqual(result["text"], "hello world")
        sent_body = json.loads(seen_bodies[0])
        self.assertTrue(sent_body["stream"])
        self.assertEqual(sent_body["stream_options"], {"include_usage": True})

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
