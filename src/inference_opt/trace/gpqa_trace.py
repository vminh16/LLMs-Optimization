from __future__ import annotations

from pathlib import Path
from typing import Sequence
import json

from inference_opt.eval.gpqa import GPQAQuestion, prompt_messages
from inference_opt.trace.loader import TraceRecord


def build_gpqa_trace_records(
    questions: Sequence[GPQAQuestion],
    *,
    schedule: Sequence[tuple[int, int]] | None = None,
    model: str,
    max_tokens: int = 16,
    timestamp_step_ms: int = 1000,
) -> list[TraceRecord]:
    if schedule is None:
        schedule = [(question.request_id, index * timestamp_step_ms) for index, question in enumerate(questions)]
    if len(schedule) < len(questions):
        raise ValueError(f"Schedule has {len(schedule)} rows for {len(questions)} questions")

    records: list[TraceRecord] = []
    for question, (request_id, timestamp_ms) in zip(questions, schedule):
        records.append(
            TraceRecord(
                request_id=request_id,
                timestamp_ms=timestamp_ms,
                workload_type="gpqa_diamond",
                body={
                    "model": model,
                    "messages": prompt_messages(question),
                    "max_tokens": max_tokens,
                    "temperature": 0,
                },
            )
        )
    return records


def write_gpqa_trace_bundle(
    questions: Sequence[GPQAQuestion],
    *,
    trace_path: Path,
    answer_key_path: Path,
    schedule: Sequence[tuple[int, int]] | None,
    model: str,
    max_tokens: int = 16,
) -> None:
    records = build_gpqa_trace_records(
        questions,
        schedule=schedule,
        model=model,
        max_tokens=max_tokens,
    )

    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(
                    {
                        "request_id": record.request_id,
                        "timestamp_ms": record.timestamp_ms,
                        "workload_type": record.workload_type,
                        "body": record.body,
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    answer_key_path.parent.mkdir(parents=True, exist_ok=True)
    answer_key = {
        "answers": [
            {"request_id": record.request_id, "correct_label": question.correct_label}
            for record, question in zip(records, questions)
        ]
    }
    answer_key_path.write_text(json.dumps(answer_key, indent=2, sort_keys=True) + "\n", encoding="utf-8")
