from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable
import json
import re
import subprocess
import time

from inference_opt.serving.sweep import BASE_COMMAND_ARGS, SweepCandidate, build_run_commands, render_compose_override


Execute = Callable[..., subprocess.CompletedProcess[str]]
MEASUREMENT_VERSION = "h0.1"


class ExperimentCommandError(RuntimeError):
    pass


def _execute_checked(stage: str, command: list[str], execute: Execute) -> subprocess.CompletedProcess[str]:
    try:
        return execute(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise ExperimentCommandError(f"{stage} failed: command not found: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or str(exc)).strip()
        raise ExperimentCommandError(f"{stage} failed: {detail}") from exc


def _parse_compose_ps(output: str) -> list[dict[str, Any]]:
    text = output.strip()
    if not text:
        return []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return payload if isinstance(payload, list) else [payload]


def _require_running_container(command: list[str], execute: Execute) -> dict[str, Any]:
    result = _execute_checked("container state check", command, execute)
    rows = _parse_compose_ps(result.stdout)
    if not rows:
        raise ExperimentCommandError("model container is missing during startup")
    model = next((row for row in rows if row.get("Service") == "model"), rows[0])
    state = str(model.get("State", "unknown")).lower()
    if state != "running":
        exit_code = model.get("ExitCode", "unknown")
        raise ExperimentCommandError(f"model container state={state}, exit_code={exit_code}")
    return model


def _monitor_startup(
    run: "ExperimentRun",
    execute: Execute,
    *,
    grace_s: float,
    poll_interval_s: float,
) -> None:
    command = _compose_args(run) + ["ps", "--all", "--format", "json", "model"]
    deadline = time.monotonic() + max(grace_s, 0.0)
    while True:
        _require_running_container(command, execute)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(max(poll_interval_s, 0.1), remaining))


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _image_from_compose(path: Path) -> str:
    match = re.search(r"^\s*image:\s*(\S+)\s*$", path.read_text(encoding="utf-8"), re.MULTILINE)
    if not match:
        raise ValueError(f"model image is missing from {path}")
    return match.group(1)


@dataclass(frozen=True)
class ExperimentRun:
    candidate: SweepCandidate
    base_compose: Path
    trace: Path
    output_dir: Path
    run_id: str
    fingerprint: str
    base_compose_sha256: str
    trace_sha256: str
    image: str

    @property
    def override_path(self) -> Path:
        return self.output_dir / "compose.override.yml"

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "experiment.json"

    @property
    def docker_log_path(self) -> Path:
        return self.output_dir / "docker.log"


def build_experiment_run(
    candidate: SweepCandidate,
    base_compose: Path,
    trace: Path,
    output_root: Path,
    run_index: int,
) -> ExperimentRun:
    base_hash = _file_sha256(base_compose)
    trace_hash = _file_sha256(trace)
    identity = {
        "candidate": candidate.name,
        "command_args": candidate.command_args,
        "base_compose_sha256": base_hash,
        "trace_sha256": trace_hash,
        "measurement_version": MEASUREMENT_VERSION,
    }
    fingerprint = sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()
    run_id = f"{candidate.name}-{run_index:02d}"
    return ExperimentRun(
        candidate=candidate,
        base_compose=base_compose,
        trace=trace,
        output_dir=output_root / run_id,
        run_id=run_id,
        fingerprint=fingerprint,
        base_compose_sha256=base_hash,
        trace_sha256=trace_hash,
        image=_image_from_compose(base_compose),
    )


def _manifest(run: ExperimentRun, status: str, **extra: Any) -> dict[str, Any]:
    result = {
        "candidate": run.candidate.name,
        "run_id": run.run_id,
        "command_args": list(run.candidate.command_args),
        "comparison_control": run.candidate.comparison_control,
        "requires_gpqa": run.candidate.requires_gpqa,
        "fingerprint": run.fingerprint,
        "image": run.image,
        "base_compose_sha256": run.base_compose_sha256,
        "trace_sha256": run.trace_sha256,
        "measurement_version": MEASUREMENT_VERSION,
        "status": status,
        "updated_at": _now(),
    }
    result.update(extra)
    return result


def _write_manifest(run: ExperimentRun, status: str, **extra: Any) -> None:
    run.manifest_path.write_text(
        json.dumps(_manifest(run, status, **extra), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def prepare_run(run: ExperimentRun, *, resume: bool = False, force: bool = False) -> bool:
    if run.output_dir.exists():
        if resume:
            if not run.manifest_path.exists():
                raise ValueError(f"cannot resume {run.run_id}: experiment.json is missing")
            manifest = json.loads(run.manifest_path.read_text(encoding="utf-8"))
            if manifest.get("fingerprint") != run.fingerprint:
                raise ValueError(f"cannot resume {run.run_id}: fingerprint does not match")
            if manifest.get("status") == "completed" and (run.output_dir / "summary.json").exists():
                _validate_summary(run)
                return True
        elif not force:
            raise FileExistsError(f"run directory already exists: {run.output_dir}")

    if force and run.output_dir.exists():
        for name in ("compose.override.yml", "experiment.json", "summary.json", "requests.jsonl", "docker.log"):
            path = run.output_dir / name
            if path.is_file():
                path.unlink()

    run.output_dir.mkdir(parents=True, exist_ok=True)
    run.override_path.write_text(render_compose_override(run.candidate), encoding="utf-8")
    _write_manifest(run, "prepared")
    return False


def _compose_args(run: ExperimentRun) -> list[str]:
    return [
        "docker",
        "compose",
        "-f",
        run.base_compose.as_posix(),
        "-f",
        run.override_path.as_posix(),
    ]


def build_preflight_commands(run: ExperimentRun) -> list[list[str]]:
    compose = _compose_args(run)
    return [
        ["docker", "info"],
        compose + ["config", "--quiet"],
        compose + ["config", "--format", "json"],
        compose + ["run", "--rm", "--no-deps", "model", "--help"],
        [
            "docker",
            "run",
            "--rm",
            "--gpus",
            "all",
            "--entrypoint",
            "python3",
            run.image,
            "-c",
            "import torch; assert torch.cuda.is_available()",
        ],
    ]


def _expected_command(run: ExperimentRun) -> list[str]:
    return [arg.split(" #", 1)[0] for arg in run.candidate.command_args]


def _validate_local_model(run: ExperimentRun) -> None:
    model_dir = run.base_compose.parent / "models" / "Qwen3.5-2B"
    required = [model_dir / "config.json", model_dir / "tokenizer_config.json"]
    missing = [path.as_posix() for path in required if not path.is_file()]
    if not list(model_dir.glob("*.safetensors")):
        missing.append(f"{model_dir.as_posix()}/*.safetensors")
    if missing:
        raise FileNotFoundError(f"local model is incomplete: {', '.join(missing)}")


def run_preflight(run: ExperimentRun, *, execute: Execute = subprocess.run) -> None:
    _validate_local_model(run)
    commands = build_preflight_commands(run)
    _execute_checked("Docker engine check", commands[0], execute)
    _execute_checked("Compose validation", commands[1], execute)
    config = _execute_checked("Compose JSON rendering", commands[2], execute)
    try:
        payload = json.loads(config.stdout)
    except json.JSONDecodeError as exc:
        detail = (config.stderr or config.stdout or "empty output").strip()
        raise ExperimentCommandError(f"Compose JSON rendering returned invalid JSON: {detail}") from exc
    model = payload["services"]["model"]
    if model.get("entrypoint") != ["python3", "-m", "vllm.entrypoints.openai.api_server"]:
        raise ValueError("resolved Compose entrypoint does not match the organizer contract")
    if model.get("command") != _expected_command(run):
        raise ValueError("resolved Compose command does not match the candidate command")
    if model.get("image") != run.image:
        raise ValueError("resolved Compose image changed unexpectedly")

    help_result = _execute_checked("vLLM CLI validation", commands[3], execute)
    help_text = f"{help_result.stdout}\n{help_result.stderr}"
    baseline_flags = {arg.split(" #", 1)[0] for arg in BASE_COMMAND_ARGS}
    candidate_flags = set(_expected_command(run)) - baseline_flags
    for arg in candidate_flags:
        if arg.split("=", 1)[0] not in help_text:
            raise ValueError(f"vLLM image does not advertise candidate flag: {arg}")
    _execute_checked("GPU probe", commands[4], execute)


def _cleanup(run: ExperimentRun, execute: Execute) -> str | None:
    command = _compose_args(run) + ["down"]
    try:
        result = execute(command, check=False, capture_output=True, text=True)
    except Exception as exc:
        return str(exc)
    if result.returncode != 0:
        return (result.stderr or result.stdout or f"exit code {result.returncode}").strip()
    return None


def _capture_diagnostics(run: ExperimentRun, execute: Execute) -> None:
    compose = _compose_args(run)
    outputs = []
    for command in (
        compose + ["ps", "--all"],
        compose + ["logs", "--no-color", "model"],
    ):
        try:
            result = execute(command, check=False, capture_output=True, text=True)
            outputs.append(f"$ {' '.join(command)}\n{result.stdout}\n{result.stderr}")
        except Exception as exc:
            outputs.append(f"$ {' '.join(command)}\n<diagnostic failed: {exc}>")
    try:
        id_command = compose + ["ps", "--all", "--quiet", "model"]
        result = execute(id_command, check=False, capture_output=True, text=True)
        outputs.append(f"$ {' '.join(id_command)}\n{result.stdout}\n{result.stderr}")
        container_id = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
        if container_id:
            for command in (["docker", "inspect", container_id], ["docker", "logs", container_id]):
                result = execute(command, check=False, capture_output=True, text=True)
                outputs.append(f"$ {' '.join(command)}\n{result.stdout}\n{result.stderr}")
    except Exception as exc:
        outputs.append(f"<direct container diagnostic failed: {exc}>")
    run.docker_log_path.write_text("\n".join(outputs), encoding="utf-8")


def _validate_summary(run: ExperimentRun) -> None:
    summary_path = run.output_dir / "summary.json"
    if not summary_path.exists():
        raise ValueError(f"benchmark did not create {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if int(summary.get("total_count", 0)) != 120 or int(summary.get("failed_count", 0)) != 0:
        raise ValueError("benchmark result must contain 120 requests and zero failures")
    if summary.get("trace_sha256") != run.trace_sha256:
        raise ValueError("benchmark trace hash does not match the experiment manifest")
    if summary.get("measurement_version") != MEASUREMENT_VERSION:
        raise ValueError("benchmark measurement version does not match the experiment manifest")


def run_experiment(
    run: ExperimentRun,
    *,
    python_executable: str,
    execute: Execute = subprocess.run,
    skip_preflight: bool = False,
    startup_grace_s: float = 60.0,
    poll_interval_s: float = 5.0,
    total_timeout_s: float = 300.0,
    stable_successes: int = 2,
) -> None:
    commands = build_run_commands(
        base_compose=run.base_compose,
        override_compose=run.override_path,
        trace=run.trace,
        output_root=run.output_dir.parent,
        run_id=run.run_id,
        python_executable=python_executable,
        startup_grace_s=0.0,
        poll_interval_s=poll_interval_s,
        total_timeout_s=total_timeout_s,
        stable_successes=stable_successes,
    )
    _write_manifest(run, "running", started_at=_now())
    primary_error: Exception | None = None
    try:
        if not skip_preflight:
            run_preflight(run, execute=execute)
        _execute_checked("initial cleanup", commands[0], execute)
        _execute_checked("model startup", commands[1], execute)
        _monitor_startup(
            run,
            execute,
            grace_s=startup_grace_s,
            poll_interval_s=poll_interval_s,
        )
        _execute_checked("model readiness", commands[2], execute)
        _execute_checked("trace benchmark", commands[3], execute)
        _validate_summary(run)
        _capture_diagnostics(run, execute)
        _write_manifest(run, "completed", completed_at=_now())
    except Exception as exc:
        primary_error = exc
        _capture_diagnostics(run, execute)
        _write_manifest(run, "failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        cleanup_error = _cleanup(run, execute)
        if cleanup_error:
            if primary_error is not None:
                _write_manifest(
                    run,
                    "failed",
                    error=f"{type(primary_error).__name__}: {primary_error}",
                    cleanup_error=cleanup_error,
                )
            else:
                _write_manifest(run, "failed", cleanup_error=cleanup_error)
                raise ExperimentCommandError(f"final cleanup failed: {cleanup_error}")
