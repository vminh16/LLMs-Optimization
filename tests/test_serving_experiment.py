import json
from pathlib import Path
from subprocess import CalledProcessError, CompletedProcess
from tempfile import TemporaryDirectory
import unittest

from inference_opt.serving.experiment import (
    build_experiment_run,
    build_preflight_commands,
    prepare_run,
    run_experiment,
)
from inference_opt.serving.sweep import select_candidates


class ServingExperimentTest(unittest.TestCase):
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
        self.assertEqual(commands[1][-2:], ["config", "--quiet"])
        self.assertEqual(commands[2][-3:], ["config", "--format", "json"])
        self.assertEqual(commands[3][-5:], ["run", "--rm", "--no-deps", "model", "--help"])
        self.assertIn("--gpus", commands[4])

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

            with self.assertRaises(CalledProcessError):
                run_experiment(run, python_executable="python", execute=execute, skip_preflight=True)

            manifest = json.loads(run.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "failed")
            self.assertTrue(run.docker_log_path.exists())
            self.assertEqual(calls[-1][-1], "down")


if __name__ == "__main__":
    unittest.main()
