from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from inference_opt.serving.sweep import (
    build_baseline_candidates,
    build_kv_fp8_batched_token_candidates,
    build_kv_fp8_seq_candidates,
    build_run_commands,
    run_commands,
    write_compose_override,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local vLLM serving config sweeps through Docker Compose.")
    parser.add_argument("--mode", choices=["baseline", "seqs", "batched-tokens"], default="baseline")
    parser.add_argument("--seqs", nargs="+", type=int, default=[32, 64, 96, 128])
    parser.add_argument("--batched-tokens", nargs="+", type=int, default=[2048, 4096, 8192])
    parser.add_argument("--fixed-max-num-seqs", type=int, default=64)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--base-compose", default="docker-compose.local.yml")
    parser.add_argument("--trace", default="data/trace-round1-diverse-content.jsonl")
    parser.add_argument("--output-root", default="results/trace-sweeps")
    parser.add_argument("--override-root", default="results/sweeps/_overrides")
    parser.add_argument("--startup-grace-s", type=float, default=60.0)
    parser.add_argument("--poll-interval-s", type=float, default=5.0)
    parser.add_argument("--total-timeout-s", type=float, default=300.0)
    parser.add_argument("--stable-successes", type=int, default=2)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode == "baseline":
        candidates = build_baseline_candidates()
    elif args.mode == "seqs":
        candidates = build_kv_fp8_seq_candidates(args.seqs)
    else:
        candidates = build_kv_fp8_batched_token_candidates(
            args.batched_tokens,
            max_num_seqs=args.fixed_max_num_seqs,
        )

    for candidate in candidates:
        override_path = write_compose_override(Path(args.override_root), candidate)
        for index in range(1, args.repeat + 1):
            run_id = f"{candidate.name}-{index:02d}"
            commands = build_run_commands(
                base_compose=Path(args.base_compose),
                override_compose=override_path,
                trace=Path(args.trace),
                output_root=Path(args.output_root),
                run_id=run_id,
                python_executable=args.python,
                startup_grace_s=args.startup_grace_s,
                poll_interval_s=args.poll_interval_s,
                total_timeout_s=args.total_timeout_s,
                stable_successes=args.stable_successes,
            )
            if args.dry_run:
                for command in commands:
                    print(subprocess.list2cmdline(command))
            else:
                print(f"running {run_id}")
                run_commands(commands)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
