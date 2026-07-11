from __future__ import annotations

from pathlib import Path
from statistics import median
from typing import Any
import json
import re


METRIC_KEYS = (
    "erc",
    "ttft_p50_ms",
    "ttft_p95_ms",
    "tbt_median_ms",
    "makespan_ms",
    "dispatch_lag_p95_ms",
    "prompt_tokens_p50",
    "prompt_tokens_p95",
    "ers",
    "final_score",
)

IDENTITY_KEYS = ("measurement_version", "trace_sha256")


def compare_candidate_to_baseline(
    baseline: dict[str, float],
    candidate: dict[str, float],
) -> dict[str, Any]:
    keys = ("tbt_median_ms", "ttft_p95_ms", "makespan_ms")
    changes = {
        key: (float(candidate[key]) / float(baseline[key]) - 1.0) * 100.0
        for key in keys
    }
    reject = (
        changes["tbt_median_ms"] >= 2.0
        or changes["makespan_ms"] >= 3.0
        or changes["ttft_p95_ms"] >= 7.0
    )
    advance = (
        (changes["tbt_median_ms"] <= -1.0 or changes["ttft_p95_ms"] <= -3.0)
        and changes["makespan_ms"] <= 2.0
    )
    decision = "reject" if reject else "advance" if advance else "inconclusive"
    return {"decision": decision, "changes_pct": changes}


def load_run_summaries(root: Path) -> list[dict[str, Any]]:
    runs = []
    for summary_path in sorted(root.glob("*/summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["run_id"] = summary_path.parent.name
        runs.append(summary)
    return runs


def _metric_summary(values: list[float]) -> dict[str, float | int]:
    center = float(median(values))
    deviations = [abs(value - center) for value in values]
    return {
        "count": len(values),
        "median": center,
        "min": min(values),
        "max": max(values),
        "mad": float(median(deviations)),
    }


def _identity_issues(runs: list[dict[str, Any]]) -> list[str]:
    issues = []
    for key in IDENTITY_KEYS:
        values = sorted({str(run[key]) if run.get(key) is not None else "<missing>" for run in runs})
        if len(values) > 1:
            issues.append(f"mixed {key} values: {', '.join(values)}")
    return issues


def _identity(runs: list[dict[str, Any]]) -> dict[str, str | None]:
    identity = {}
    for key in IDENTITY_KEYS:
        values = {str(run[key]) for run in runs if run.get(key) is not None}
        identity[key] = next(iter(values)) if len(values) == 1 else None
    return identity


def summarize_run_group(
    root: Path,
    *,
    min_runs: int,
    expected_total_count: int,
) -> dict[str, Any]:
    runs = load_run_summaries(root)
    issues = []

    if len(runs) < min_runs:
        issues.append(f"expected at least {min_runs} runs, found {len(runs)}")

    for run in runs:
        run_id = run["run_id"]
        failed_count = int(run.get("failed_count", run.get("error_count", 0)) or 0)
        total_count = int(run.get("total_count", run.get("request_count", 0)) or 0)
        if failed_count > 0:
            issues.append(f"{run_id} has failed_count={failed_count}")
        if total_count != expected_total_count:
            issues.append(f"{run_id} has total_count={total_count}, expected {expected_total_count}")

    issues.extend(_identity_issues(runs))

    metrics = {}
    for key in METRIC_KEYS:
        values = [float(run[key]) for run in runs if run.get(key) is not None]
        if values:
            metrics[key] = _metric_summary(values)

    return {
        "root": root.as_posix(),
        "run_count": len(runs),
        "run_ids": [run["run_id"] for run in runs],
        "min_runs": min_runs,
        "expected_total_count": expected_total_count,
        "ready_for_comparison": not issues,
        "blocking_issues": issues,
        "identity": _identity(runs),
        "metrics": metrics,
    }


def _candidate_name(run_id: str) -> str:
    return re.sub(r"-\d+$", "", run_id)


def summarize_candidate_groups(
    root: Path,
    *,
    min_runs: int,
    expected_total_count: int,
) -> dict[str, Any]:
    runs = load_run_summaries(root)
    groups: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        groups.setdefault(_candidate_name(str(run["run_id"])), []).append(run)

    summaries = {}
    for name, group_runs in sorted(groups.items()):
        group_root = root / name
        run_ids = {run["run_id"] for run in group_runs}
        summary = summarize_run_group(root, min_runs=min_runs, expected_total_count=expected_total_count)
        summary["root"] = group_root.as_posix()
        summary["run_count"] = len(group_runs)
        summary["run_ids"] = sorted(run_ids)
        summary["blocking_issues"] = [
            issue for issue in summary["blocking_issues"] if any(run_id in issue for run_id in run_ids)
        ]
        summary["blocking_issues"].extend(_identity_issues(group_runs))
        if len(group_runs) < min_runs:
            summary["blocking_issues"].insert(0, f"expected at least {min_runs} runs, found {len(group_runs)}")

        metrics = {}
        for key in METRIC_KEYS:
            values = [float(run[key]) for run in group_runs if run.get(key) is not None]
            if values:
                metrics[key] = _metric_summary(values)
        summary["metrics"] = metrics
        summary["identity"] = _identity(group_runs)
        summary["ready_for_comparison"] = not summary["blocking_issues"]
        summaries[name] = summary

    return {
        "root": root.as_posix(),
        "group_count": len(summaries),
        "groups": summaries,
    }
