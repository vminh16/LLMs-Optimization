from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json


@dataclass(frozen=True)
class TraceRecord:
    request_id: int
    timestamp_ms: int
    workload_type: str
    body: dict[str, Any]


def load_trace(path: Path) -> list[TraceRecord]:
    records: list[TraceRecord] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        raw = json.loads(line)
        records.append(
            TraceRecord(
                request_id=int(raw["request_id"]),
                timestamp_ms=int(raw["timestamp_ms"]),
                workload_type=str(raw["workload_type"]),
                body=dict(raw["body"]),
            )
        )
    return records
