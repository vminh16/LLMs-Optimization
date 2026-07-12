from __future__ import annotations

import argparse
import json
from pathlib import Path

from inference_opt.benchmark.report import (
    compare_candidate_to_baseline,
    summarize_candidate_groups,
    summarize_run_group,
)
from inference_opt.serving.sweep import build_experiment1_candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Experiment 1 candidates with the locked H0.1 baseline.")
    parser.add_argument("--baseline-root", default="results/trace-baseline-h01")
    parser.add_argument("--experiment-root", default="results/experiment1")
    parser.add_argument("--min-runs", type=int, default=1)
    parser.add_argument("--expected-total-count", type=int, default=120)
    parser.add_argument("--output")
    return parser.parse_args()


def _medians(summary: dict) -> dict[str, float]:
    return {
        key: float(summary["metrics"][key]["median"])
        for key in ("tbt_median_ms", "ttft_p95_ms", "makespan_ms")
    }


def build_report(args: argparse.Namespace) -> dict:
    baseline = summarize_run_group(
        Path(args.baseline_root), min_runs=3, expected_total_count=args.expected_total_count
    )
    candidates = summarize_candidate_groups(
        Path(args.experiment_root),
        min_runs=args.min_runs,
        expected_total_count=args.expected_total_count,
        require_manifests=True,
    )
    comparisons = {}
    specs = {candidate.name: candidate for candidate in build_experiment1_candidates()}
    if baseline["ready_for_comparison"]:
        for name, candidate in candidates["groups"].items():
            if not candidate["ready_for_comparison"]:
                comparisons[name] = {"decision": "blocked", "blocking_issues": candidate["blocking_issues"]}
                continue
            spec = specs.get(name)
            control_name = spec.comparison_control if spec else "baseline"
            control = baseline if control_name == "baseline" else candidates["groups"].get(control_name)
            if control is None or not control["ready_for_comparison"]:
                issues = [] if control is None else control["blocking_issues"]
                comparisons[name] = {
                    "decision": "blocked",
                    "comparison_control": control_name,
                    "blocking_issues": [f"control {control_name} is not ready", *issues],
                }
                continue
            if candidate["identity"] != control["identity"]:
                comparisons[name] = {
                    "decision": "blocked",
                    "comparison_control": control_name,
                    "blocking_issues": ["candidate trace or measurement version differs from control"],
                }
                continue
            comparison = compare_candidate_to_baseline(_medians(control), _medians(candidate))
            comparison["comparison_control"] = control_name
            if spec and spec.requires_gpqa and comparison["decision"] == "advance":
                comparison["accuracy_status"] = "provisional_pending_gpqa"
            comparisons[name] = comparison
    return {"baseline": baseline, "candidates": candidates, "comparisons": comparisons}


def main() -> int:
    args = parse_args()
    payload = json.dumps(build_report(args), indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
