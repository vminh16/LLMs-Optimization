from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
from pathlib import Path

from inference_opt.benchmark.runner import BenchmarkConfig, run_benchmark, write_benchmark_result
from inference_opt.trace.loader import load_trace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay a JSONL trace against an OpenAI-compatible vLLM server.")
    parser.add_argument("--trace", default="data/trace-round1.jsonl")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--request-timeout-s", type=float, default=120.0)
    parser.add_argument("--output-root", default="results/baseline")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--no-respect-timestamps", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(args.output_root) / run_id
    records = load_trace(Path(args.trace))
    result = asyncio.run(
        run_benchmark(
            records,
            BenchmarkConfig(
                base_url=args.base_url,
                request_timeout_s=args.request_timeout_s,
            ),
            respect_timestamps=not args.no_respect_timestamps,
        )
    )
    write_benchmark_result(result, output_dir)
    print(output_dir.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
