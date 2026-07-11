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
    command_args: tuple[str, ...]


def build_baseline_candidates() -> list[SweepCandidate]:
    return [SweepCandidate(name="baseline-cold", command_args=BASE_COMMAND_ARGS)]


def _replace_prefix_flag(replacement: str) -> tuple[str, ...]:
    return tuple(arg for arg in BASE_COMMAND_ARGS if arg != "--enable-prefix-caching") + (replacement,)


def validate_command_args(command_args: tuple[str, ...]) -> None:
    keys = []
    for arg in command_args:
        key = arg.split(" #", 1)[0].split("=", 1)[0]
        if key in keys:
            raise ValueError(f"duplicate vLLM flag: {key}")
        keys.append(key)
    if "--enable-prefix-caching" in keys and "--no-enable-prefix-caching" in keys:
        raise ValueError("conflicting prefix caching flags")


def build_experiment1_candidates() -> list[SweepCandidate]:
    candidates = [
        SweepCandidate("language-only", BASE_COMMAND_ARGS + ("--language-model-only",)),
        SweepCandidate("renderer-2", BASE_COMMAND_ARGS + ("--renderer-num-workers=2",)),
        SweepCandidate(
            "performance-interactivity",
            BASE_COMMAND_ARGS + ("--performance-mode=interactivity",),
        ),
        SweepCandidate(
            "performance-throughput",
            BASE_COMMAND_ARGS + ("--performance-mode=throughput",),
        ),
        SweepCandidate("prefix-off", _replace_prefix_flag("--no-enable-prefix-caching")),
    ]
    for candidate in candidates:
        validate_command_args(candidate.command_args)
    return candidates


def select_candidates(names: list[str] | None = None) -> list[SweepCandidate]:
    candidates = build_experiment1_candidates()
    if not names:
        return candidates
    by_name = {candidate.name: candidate for candidate in candidates}
    unknown = [name for name in names if name not in by_name]
    if unknown:
        raise ValueError(f"unknown Experiment 1 candidate: {', '.join(unknown)}")
    return [by_name[name] for name in names]


def render_compose_override(candidate: SweepCandidate) -> str:
    validate_command_args(candidate.command_args)
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
    compose_args = ["docker", "compose", "-f", _path(base_compose), "-f", _path(override_compose)]
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
