import json
from pathlib import Path
from subprocess import CalledProcessError, CompletedProcess
from tempfile import TemporaryDirectory
import unittest

from inference_opt.serving.experiment import (
    ExperimentCommandError,
    _execute_checked,
    build_experiment_run,
    build_preflight_commands,
    prepare_run,
    run_experiment,
)
from inference_opt.serving.sweep import select_candidates


class ServingExperimentTest(unittest.TestCase):
    def test_checked_command_includes_stage_and_stderr(self):
        def execute(command, **kwargs):
            raise CalledProcessError(2, command, output="stdout detail", stderr="stderr detail")

        with self.assertRaisesRegex(ExperimentCommandError, "GPU probe.*stderr detail"):
            _execute_checked("GPU probe", ["docker", "run"], execute)

    def setUp(self):
        self.candidate = select_candidates(["renderer-2"])[0]

    def _inputs(self, root: Path):
        compose = root / "docker-compose.local.yml"
        trace = root / "trace.jsonl"
        compose.write_text("services:\n  model:\n    image: vllm/vllm-openai:v0.22.1\n", encoding="utf-8")
        trace.write_text('{"request_id":"r1"}\n', encoding="utf-8")
        return compose, trace

    def test_fingerprint_is_deterministic_and_changes_with_command(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            compose, trace = self._inputs(root)
            first = build_experiment_run(self.candidate, compose, trace, root / "results", 1)
            second = build_experiment_run(self.candidate, compose, trace, root / "results", 1)
            other = build_experiment_run(select_candidates(["language-only"])[0], compose, trace, root / "results", 1)

        self.assertEqual(first.fingerprint, second.fingerprint)
        self.assertNotEqual(first.fingerprint, other.fingerprint)

    def test_prepare_refuses_existing_run_without_resume_or_force(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            compose, trace = self._inputs(root)
            run = build_experiment_run(self.candidate, compose, trace, root / "results", 1)
            run.output_dir.mkdir(parents=True)

            with self.assertRaisesRegex(FileExistsError, "already exists"):
                prepare_run(run)

    def test_resume_skips_only_completed_matching_run(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            compose, trace = self._inputs(root)
            run = build_experiment_run(self.candidate, compose, trace, root / "results", 1)
            run.output_dir.mkdir(parents=True)
            (run.output_dir / "experiment.json").write_text(
                json.dumps({"fingerprint": run.fingerprint, "status": "completed"}), encoding="utf-8"
            )
            (run.output_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "total_count": 120,
                        "failed_count": 0,
                        "trace_sha256": run.trace_sha256,
                        "measurement_version": "h0.1",
                    }
                ),
                encoding="utf-8",
            )

            self.assertTrue(prepare_run(run, resume=True))

    def test_resume_rejects_completed_run_with_invalid_summary(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            compose, trace = self._inputs(root)
            run = build_experiment_run(self.candidate, compose, trace, root / "results", 1)
            run.output_dir.mkdir(parents=True)
            (run.output_dir / "experiment.json").write_text(
                json.dumps({"fingerprint": run.fingerprint, "status": "completed"}), encoding="utf-8"
            )
            (run.output_dir / "summary.json").write_text("{}", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "120 requests"):
                prepare_run(run, resume=True)

    def test_resume_rejects_different_fingerprint(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            compose, trace = self._inputs(root)
            run = build_experiment_run(self.candidate, compose, trace, root / "results", 1)
            run.output_dir.mkdir(parents=True)
            (run.output_dir / "experiment.json").write_text(
                json.dumps({"fingerprint": "different", "status": "completed"}), encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "fingerprint"):
                prepare_run(run, resume=True)

    def test_force_removes_known_stale_artifacts_before_retry(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            compose, trace = self._inputs(root)
            run = build_experiment_run(self.candidate, compose, trace, root / "results", 1)
            run.output_dir.mkdir(parents=True)
            stale_summary = run.output_dir / "summary.json"
            stale_summary.write_text('{"total_count":120}', encoding="utf-8")

            prepare_run(run, force=True)

            self.assertFalse(stale_summary.exists())
            self.assertTrue(run.manifest_path.exists())

    def test_preflight_checks_engine_compose_help_and_gpu(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            compose, trace = self._inputs(root)
            run = build_experiment_run(self.candidate, compose, trace, root / "results", 1)
            run.output_dir.mkdir(parents=True)
            run.override_path.write_text("services: {}\n", encoding="utf-8")

            commands = build_preflight_commands(run)

        self.assertEqual(commands[0], ["docker", "info"])
        self.assertEqual(commands[1][:2], ["docker", "compose"])
        self.assertEqual(commands[1][-2:], ["config", "--quiet"])
        self.assertEqual(commands[2][-3:], ["config", "--format", "json"])
        self.assertEqual(commands[3][-5:], ["run", "--rm", "--no-deps", "model", "--help"])
        self.assertIn("--gpus", commands[4])
        self.assertEqual(
            commands[4],
            [
                "docker",
                "run",
                "--rm",
                "--gpus",
                "all",
                "--entrypoint",
                "python3",
                "vllm/vllm-openai:v0.22.1",
                "-c",
                "import torch; assert torch.cuda.is_available()",
            ],
        )

    def test_failure_captures_diagnostics_and_always_cleans_up(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            compose, trace = self._inputs(root)
            run = build_experiment_run(self.candidate, compose, trace, root / "results", 1)
            prepare_run(run)
            calls = []

            def execute(command, **kwargs):
                calls.append(command)
                if command[-3:] == ["up", "-d", "model"]:
                    raise CalledProcessError(1, command, output="up failed")
                return CompletedProcess(command, 0, stdout="diagnostic output", stderr="")

            with self.assertRaisesRegex(ExperimentCommandError, "model startup.*up failed"):
                run_experiment(run, python_executable="python", execute=execute, skip_preflight=True)

            manifest = json.loads(run.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "failed")
            self.assertTrue(run.docker_log_path.exists())
            self.assertEqual(calls[-1][-1], "down")

    def test_initial_cleanup_failure_stops_before_model_start(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            compose, trace = self._inputs(root)
            run = build_experiment_run(self.candidate, compose, trace, root / "results", 1)
            prepare_run(run)
            calls = []

            def execute(command, **kwargs):
                calls.append(command)
                if command[-1] == "down":
                    raise CalledProcessError(1, command, stderr="cleanup failed")
                return CompletedProcess(command, 0, stdout="", stderr="")

            with self.assertRaisesRegex(ExperimentCommandError, "initial cleanup"):
                run_experiment(run, python_executable="python", execute=execute, skip_preflight=True)

            self.assertFalse(any(command[-3:] == ["up", "-d", "model"] for command in calls))

    def test_successful_run_completes_manifest_after_cleanup(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            compose, trace = self._inputs(root)
            run = build_experiment_run(self.candidate, compose, trace, root / "results", 1)
            prepare_run(run)

            def execute(command, **kwargs):
                if "scripts/run_trace_benchmark.py" in command:
                    (run.output_dir / "summary.json").write_text(
                        json.dumps(
                            {
                                "total_count": 120,
                                "failed_count": 0,
                                "trace_sha256": run.trace_sha256,
                                "measurement_version": "h0.1",
                            }
                        ),
                        encoding="utf-8",
                    )
                return CompletedProcess(command, 0, stdout="", stderr="")

            run_experiment(run, python_executable="python", execute=execute, skip_preflight=True)

            manifest = json.loads(run.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "completed")

    def test_final_cleanup_failure_marks_successful_run_failed(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            compose, trace = self._inputs(root)
            run = build_experiment_run(self.candidate, compose, trace, root / "results", 1)
            prepare_run(run)
            down_count = 0

            def execute(command, **kwargs):
                nonlocal down_count
                if command[-1] == "down":
                    down_count += 1
                    if down_count == 2:
                        return CompletedProcess(command, 1, stdout="", stderr="cleanup failed")
                if "scripts/run_trace_benchmark.py" in command:
                    (run.output_dir / "summary.json").write_text(
                        json.dumps(
                            {
                                "total_count": 120,
                                "failed_count": 0,
                                "trace_sha256": run.trace_sha256,
                                "measurement_version": "h0.1",
                            }
                        ),
                        encoding="utf-8",
                    )
                return CompletedProcess(command, 0, stdout="", stderr="")

            with self.assertRaisesRegex(ExperimentCommandError, "final cleanup.*cleanup failed"):
                run_experiment(run, python_executable="python", execute=execute, skip_preflight=True)

            manifest = json.loads(run.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["cleanup_error"], "cleanup failed")


if __name__ == "__main__":
    unittest.main()
