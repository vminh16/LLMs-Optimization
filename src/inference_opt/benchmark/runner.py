from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Awaitable, Callable
import asyncio
import json

import httpx

from inference_opt.benchmark.scoring import RequestMeasurement, ScoreConfig, request_score, summarize_scores
from inference_opt.trace.loader import TraceRecord


Sender = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


@dataclass(frozen=True)
class BenchmarkConfig:
    base_url: str
    request_timeout_s: float
    score_config: ScoreConfig = ScoreConfig()


@dataclass(frozen=True)
class BenchmarkResult:
    requests: list[dict[str, Any]]
    summary: dict[str, Any]


def _chat_completions_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/chat/completions"


def _content_from_chunk(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        return ""
    choice = choices[0]
    delta = choice.get("delta") or {}
    return str(delta.get("content") or delta.get("reasoning_content") or choice.get("text") or "")


def _completion_tokens_from_chunk(payload: dict[str, Any]) -> int | None:
    usage = payload.get("usage") or {}
    completion_tokens = usage.get("completion_tokens")
    return int(completion_tokens) if completion_tokens is not None else None


async def send_openai_streaming_request(
    body: dict[str, Any],
    *,
    base_url: str,
    timeout_s: float,
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    request_body = dict(body)
    request_body["stream"] = True
    stream_options = dict(request_body.get("stream_options") or {})
    stream_options["include_usage"] = True
    request_body["stream_options"] = stream_options

    chunks: list[str] = []
    token_times: list[float] = []
    usage_completion_tokens: int | None = None
    started = perf_counter()

    owns_client = client is None
    if client is None:
        client = httpx.AsyncClient(timeout=httpx.Timeout(timeout_s))

    try:
        async with client.stream("POST", _chat_completions_url(base_url), json=request_body) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line.removeprefix("data: ").strip()
                if data == "[DONE]":
                    break
                payload = json.loads(data)
                usage_completion_tokens = _completion_tokens_from_chunk(payload) or usage_completion_tokens
                content = _content_from_chunk(payload)
                if content:
                    token_times.append(perf_counter())
                    chunks.append(content)
    finally:
        if owns_client:
            await client.aclose()

    finished = perf_counter()
    total_ms = (finished - started) * 1000.0
    output_tokens = usage_completion_tokens if usage_completion_tokens is not None else len(chunks)
    ttft_ms = (token_times[0] - started) * 1000.0 if token_times else None
    if output_tokens > 1 and ttft_ms is not None:
        tpot_ms = (total_ms - ttft_ms) / (output_tokens - 1)
    elif output_tokens == 1:
        tpot_ms = 0.0
    else:
        tpot_ms = None

    return {
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "output_tokens": output_tokens,
        "total_ms": total_ms,
        "text": "".join(chunks),
    }


async def _run_one(
    record: TraceRecord,
    config: BenchmarkConfig,
    sender: Sender,
    *,
    delay_s: float,
) -> tuple[dict[str, Any], RequestMeasurement]:
    if delay_s > 0:
        await asyncio.sleep(delay_s)

    try:
        raw = await sender(record.body)
        measurement = RequestMeasurement(
            ttft_ms=raw.get("ttft_ms"),
            tpot_ms=raw.get("tpot_ms"),
            output_tokens=int(raw.get("output_tokens") or 0),
        )
        error = None
    except Exception as exc:
        raw = {}
        measurement = RequestMeasurement(error=str(exc), output_tokens=0)
        error = str(exc)

    score = request_score(measurement, config.score_config)
    row = {
        "request_id": record.request_id,
        "timestamp_ms": record.timestamp_ms,
        "workload_type": record.workload_type,
        "ttft_ms": measurement.ttft_ms,
        "tpot_ms": measurement.tpot_ms,
        "output_tokens": measurement.output_tokens,
        "score": score,
        "error": error,
    }
    if "total_ms" in raw:
        row["total_ms"] = raw["total_ms"]
    if "text" in raw:
        row["text"] = raw["text"]
    return row, measurement


async def run_benchmark(
    records: list[TraceRecord],
    config: BenchmarkConfig,
    *,
    sender: Sender | None = None,
    respect_timestamps: bool = True,
) -> BenchmarkResult:
    if sender is None:
        async with httpx.AsyncClient(timeout=httpx.Timeout(config.request_timeout_s)) as client:
            sender = lambda body: send_openai_streaming_request(
                body,
                base_url=config.base_url,
                timeout_s=config.request_timeout_s,
                client=client,
            )
            return await run_benchmark(
                records,
                config,
                sender=sender,
                respect_timestamps=respect_timestamps,
            )

    first_timestamp = min((record.timestamp_ms for record in records), default=0)
    tasks = []
    for record in records:
        delay_s = ((record.timestamp_ms - first_timestamp) / 1000.0) if respect_timestamps else 0.0
        tasks.append(_run_one(record, config, sender, delay_s=delay_s))

    pairs = await asyncio.gather(*tasks)
    rows = [row for row, _measurement in pairs]
    measurements = [measurement for _row, measurement in pairs]
    summary = summarize_scores(measurements, config.score_config)
    return BenchmarkResult(requests=rows, summary=summary)


def write_benchmark_result(result: BenchmarkResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(result.summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "requests.jsonl").open("w", encoding="utf-8") as file:
        for row in result.requests:
            file.write(json.dumps(row, sort_keys=True) + "\n")
