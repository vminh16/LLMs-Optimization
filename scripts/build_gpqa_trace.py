from __future__ import annotations

import argparse
import os
from pathlib import Path

from inference_opt.eval.gpqa import DATASET_ID, load_gpqa_rows, sample_gpqa_questions
from inference_opt.serving.model_download import load_env_file
from inference_opt.trace.gpqa_trace import write_gpqa_trace_bundle
from inference_opt.trace.loader import load_trace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a 120-request GPQA Diamond trace for local benchmarking.")
    parser.add_argument("--env-file", default=".env", help="Local env file with HF_TOKEN and GPQA settings.")
    parser.add_argument("--dataset-id", default=None)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--count", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--schedule-trace", default="data/trace-round1.jsonl")
    parser.add_argument("--output-trace", default="data/traces/gpqa-diamond-120.jsonl")
    parser.add_argument("--output-answer-key", default="data/traces/gpqa-diamond-120.answers.json")
    return parser.parse_args()


def resolve_hf_token(values: dict[str, str]) -> str | None:
    return values.get("HF_TOKEN") or os.getenv("HF_TOKEN") or None


def main() -> int:
    args = parse_args()
    values = load_env_file(Path(args.env_file))
    token = resolve_hf_token(values)
    try:
        rows = load_gpqa_rows(
            dataset_id=args.dataset_id or values.get("GPQA_DATASET_ID") or os.getenv("GPQA_DATASET_ID") or DATASET_ID,
            variant=args.variant or values.get("GPQA_VARIANT") or os.getenv("GPQA_VARIANT") or "diamond",
            split=args.split,
            token=token,
        )
    except Exception as exc:
        raise SystemExit(
            "Could not load GPQA Diamond. Set HF_TOKEN after accepting the dataset terms, "
            f"then rerun this command. Error: {exc}"
        ) from exc

    questions = sample_gpqa_questions(rows, count=args.count, seed=args.seed)
    schedule_records = load_trace(Path(args.schedule_trace))
    schedule = [(record.request_id, record.timestamp_ms) for record in schedule_records[: args.count]]

    write_gpqa_trace_bundle(
        questions,
        trace_path=Path(args.output_trace),
        answer_key_path=Path(args.output_answer_key),
        schedule=schedule,
        model=args.model or values.get("MODEL_SERVED_NAME") or os.getenv("MODEL_SERVED_NAME") or "Qwen3.5-2B",
        max_tokens=args.max_tokens,
    )
    print(Path(args.output_trace).as_posix())
    print(Path(args.output_answer_key).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
