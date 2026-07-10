from __future__ import annotations

import argparse
import json
from pathlib import Path

from inference_opt.benchmark.report import summarize_candidate_groups, summarize_run_group


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize repeated trace benchmark runs.")
    parser.add_argument("--root", default="results/trace-baseline-h0")
    parser.add_argument("--min-runs", type=int, default=3)
    parser.add_argument("--expected-total-count", type=int, default=120)
    parser.add_argument("--group-by-candidate", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.group_by_candidate:
        report = summarize_candidate_groups(
            Path(args.root),
            min_runs=args.min_runs,
            expected_total_count=args.expected_total_count,
        )
    else:
        report = summarize_run_group(
            Path(args.root),
            min_runs=args.min_runs,
            expected_total_count=args.expected_total_count,
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
