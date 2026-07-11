from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess


BASE_COMMAND_ARGS = (
    "--model=/model #Don't change this to vllm-server",
    "--served-model-name=Qwen3.5-2B #Don't change this to vllm-server",
    "--host=0.0.0.0 #Don't change this to vllm-server",
    "--port=8000 #Don't change this to vllm-server",
    "--max-model-len=262144",
    "--gpu-memory-utilization=0.95",
    "--tensor-parallel-size=1",
    "--enable-prefix-caching",
)


@dataclass(frozen=True)
class SweepCandidate:
    name: str
    extra_args: tuple[str, ...]

    @property
    def command_args(self) -> tuple[str, ...]:
        return BASE_COMMAND_ARGS + self.extra_args


def build_baseline_candidates() -> list[SweepCandidate]:
    return [SweepCandidate(name="baseline-cold", extra_args=())]


def build_kv_fp8_seq_candidates(values: list[int]) -> list[SweepCandidate]:
    return [
        SweepCandidate(
            name=f"kv-fp8-seqs-{value}",
            extra_args=(
                "--kv-cache-dtype=fp8",
                "--calculate-kv-scales",
                f"--max-num-seqs={value}",
            ),
        )
        for value in values
    ]


def build_kv_fp8_batched_token_candidates(values: list[int], *, max_num_seqs: int) -> list[SweepCandidate]:
    return [
        SweepCandidate(
            name=f"kv-fp8-seqs-{max_num_seqs}-btokens-{value}",
            extra_args=(
                "--kv-cache-dtype=fp8",
                "--calculate-kv-scales",
                f"--max-num-seqs={max_num_seqs}",
                f"--max-num-batched-tokens={value}",
            ),
        )
        for value in values
    ]


def render_compose_override(candidate: SweepCandidate) -> str:
    lines = [
        "services:",
        "  model:",
        "    command:",
    ]
    lines.extend(f"      - {arg}" for arg in candidate.command_args)
    return "\n".join(lines) + "\n"


def write_compose_override(root: Path, candidate: SweepCandidate) -> Path:
    output_dir = root / candidate.name
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "compose.override.yml"
    path.write_text(render_compose_override(candidate), encoding="utf-8")
    return path


def _path(path: Path) -> str:
    return path.as_posix()


def build_run_commands(
    *,
    base_compose: Path,
    override_compose: Path,
    trace: Path,
    output_root: Path,
    run_id: str,
    python_executable: str,
    startup_grace_s: float = 60.0,
    poll_interval_s: float = 5.0,
    total_timeout_s: float = 300.0,
    stable_successes: int = 2,
) -> list[list[str]]:
    compose_args = ["docker-compose", "-f", _path(base_compose), "-f", _path(override_compose)]
    return [
        compose_args + ["down"],
        compose_args + ["up", "-d", "model"],
        [
            python_executable,
            "scripts/check_server_health.py",
            "--wait",
            "--startup-grace-s",
            str(startup_grace_s),
            "--poll-interval-s",
            str(poll_interval_s),
            "--total-timeout-s",
            str(total_timeout_s),
            "--stable-successes",
            str(stable_successes),
        ],
        [
            python_executable,
            "scripts/run_trace_benchmark.py",
            "--trace",
            _path(trace),
            "--output-root",
            _path(output_root),
            "--run-id",
            run_id,
        ],
        compose_args + ["down"],
    ]


def run_commands(commands: list[list[str]]) -> None:
    subprocess.run(commands[0], check=False)
    subprocess.run(commands[1], check=True)
    try:
        subprocess.run(commands[2], check=True)
        subprocess.run(commands[3], check=True)
    finally:
        subprocess.run(commands[4], check=False)
