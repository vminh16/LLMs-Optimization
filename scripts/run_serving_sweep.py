from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from inference_opt.serving.experiment import (
    build_experiment_run,
    build_preflight_commands,
    prepare_run,
    run_experiment,
    run_preflight,
)
from inference_opt.serving.sweep import (
    build_run_commands,
    select_candidates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local vLLM serving config sweeps through Docker Compose.")
    parser.add_argument("--suite", choices=["experiment1"], default="experiment1")
    parser.add_argument("--candidate", action="append")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--base-compose", default="docker-compose.local.yml")
    parser.add_argument("--trace", default="data/trace-round1-diverse-content.jsonl")
    parser.add_argument("--output-root", default="results/experiment1")
    parser.add_argument("--startup-grace-s", type=float, default=60.0)
    parser.add_argument("--poll-interval-s", type=float, default=5.0)
    parser.add_argument("--total-timeout-s", type=float, default=300.0)
    parser.add_argument("--stable-successes", type=int, default=2)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def run() -> int:
    args = parse_args()
    candidates = select_candidates(args.candidate)
    base_compose = Path(args.base_compose)
    trace = Path(args.trace)

    if args.preflight_only:
        with TemporaryDirectory(prefix="vllm-experiment1-preflight-") as tmp:
            for candidate in candidates:
                run = build_experiment_run(candidate, base_compose, trace, Path(tmp), 1)
                prepare_run(run)
                if args.dry_run:
                    for command in build_preflight_commands(run):
                        print(subprocess.list2cmdline(command))
                else:
                    print(f"preflight {candidate.name}")
                    run_preflight(run)
        return 0

    for candidate in candidates:
        for index in range(1, args.repeat + 1):
            run = build_experiment_run(candidate, base_compose, trace, Path(args.output_root), index)
            commands = build_run_commands(
                base_compose=base_compose,
                override_compose=run.override_path,
                trace=trace,
                output_root=Path(args.output_root),
                run_id=run.run_id,
                python_executable=args.python,
                startup_grace_s=args.startup_grace_s,
                poll_interval_s=args.poll_interval_s,
                total_timeout_s=args.total_timeout_s,
                stable_successes=args.stable_successes,
            )
            if args.dry_run:
                for command in build_preflight_commands(run):
                    print(subprocess.list2cmdline(command))
                for command in commands:
                    print(subprocess.list2cmdline(command))
            else:
                if prepare_run(run, resume=args.resume, force=args.force):
                    print(f"skipping completed {run.run_id}")
                    continue
                print(f"running {run.run_id}")
                run_experiment(
                    run,
                    python_executable=args.python,
                    startup_grace_s=args.startup_grace_s,
                    poll_interval_s=args.poll_interval_s,
                    total_timeout_s=args.total_timeout_s,
                    stable_successes=args.stable_successes,
                )
    return 0


def main() -> int:
    try:
        return run()
    except (FileExistsError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
