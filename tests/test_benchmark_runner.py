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
                b'data: {"choices":[],"usage":{"prompt_tokens":17,"completion_tokens":5}}\n\n',
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
        self.assertEqual(result["prompt_tokens"], 17)
        self.assertAlmostEqual(result["ttft_ms"], 200.0)
        self.assertAlmostEqual(result["tpot_ms"], 250.0)
        self.assertEqual(result["text"], "hello world")
        sent_body = json.loads(seen_bodies[0])
        self.assertTrue(sent_body["stream"])
        self.assertEqual(sent_body["stream_options"], {"include_usage": True})

    def test_streaming_request_rejects_missing_usage(self):
        async def handler(_request):
            lines = [
                b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n',
                b"data: [DONE]\n\n",
            ]
            return httpx.Response(200, content=b"".join(lines))

        transport = httpx.MockTransport(handler)
        clock = iter([10.0, 10.2, 10.4])

        async def run():
            async with httpx.AsyncClient(transport=transport) as client:
                return await send_openai_streaming_request(
                    {"messages": [{"role": "user", "content": "hi"}]},
                    base_url="http://test/v1",
                    timeout_s=1.0,
                    client=client,
                )

        with patch("inference_opt.benchmark.runner.perf_counter", side_effect=lambda: next(clock)):
            with self.assertRaisesRegex(ValueError, "usage"):
                asyncio.run(run())

    def test_run_benchmark_sizes_http_client_to_trace(self):
        seen_limits = []

        class FakeAsyncClient:
            def __init__(self, *, timeout, limits=None):
                self.timeout = timeout
                seen_limits.append(limits)

            async def __aenter__(self):
                return self

            async def __aexit__(self, _exc_type, _exc, _traceback):
                return None

        async def fake_streaming_request(_body, *, base_url, timeout_s, client):
            return {
                "ttft_ms": 100.0,
                "tpot_ms": 20.0,
                "prompt_tokens": 10,
                "output_tokens": 2,
                "text": "ok",
            }

        records = [
            TraceRecord(
                request_id=index,
                timestamp_ms=0,
                workload_type="conversation",
                body={"messages": [{"role": "user", "content": "ok"}]},
            )
            for index in range(120)
        ]

        with (
            patch("inference_opt.benchmark.runner.httpx.AsyncClient", FakeAsyncClient),
            patch(
                "inference_opt.benchmark.runner.send_openai_streaming_request",
                side_effect=fake_streaming_request,
            ),
        ):
            result = asyncio.run(
                run_benchmark(
                    records,
                    BenchmarkConfig(base_url="http://localhost:8000/v1", request_timeout_s=1.0),
                    respect_timestamps=False,
                )
            )

        self.assertEqual(result.summary["request_count"], 120)
        self.assertEqual(seen_limits[0].max_connections, 120)

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

    def test_run_benchmark_records_schedule_and_makespan(self):
        async def sender(_body):
            return {
                "ttft_ms": 100.0,
                "tpot_ms": 20.0,
                "prompt_tokens": 10,
                "output_tokens": 2,
                "total_ms": 120.0,
                "text": "ok",
            }

        records = [
            TraceRecord(
                request_id=1,
                timestamp_ms=0,
                workload_type="conversation",
                body={"messages": [{"role": "user", "content": "ok"}]},
            )
        ]

        result = asyncio.run(
            run_benchmark(
                records,
                BenchmarkConfig(base_url="http://localhost:8000/v1", request_timeout_s=1.0),
                sender=sender,
                respect_timestamps=True,
            )
        )

        row = result.requests[0]
        self.assertEqual(row["scheduled_offset_ms"], 0.0)
        self.assertGreaterEqual(row["dispatch_offset_ms"], 0.0)
        self.assertEqual(row["dispatch_lag_ms"], row["dispatch_offset_ms"])
        self.assertGreaterEqual(result.summary["makespan_ms"], 0.0)
        self.assertEqual(result.summary["measurement_version"], "h0.1")


if __name__ == "__main__":
    unittest.main()
