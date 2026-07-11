from pathlib import Path
import io
import shlex
import unittest
from unittest.mock import call, patch

from inference_opt.serving import sweep
from inference_opt.serving.sweep import build_run_commands, render_compose_override
from scripts import run_serving_sweep


class ServingSweepTest(unittest.TestCase):
    def test_posix_dry_run_uses_shell_safe_quoting(self):
        command = ["python", "-c", "print('x y')"]

        rendered = run_serving_sweep.format_command(command, os_name="posix")

        self.assertEqual(shlex.split(rendered), command)

    def test_main_prints_concise_runtime_error_without_traceback(self):
        stderr = io.StringIO()
        with patch.object(run_serving_sweep, "run", side_effect=RuntimeError("Start Docker Desktop")):
            with patch("sys.stderr", stderr):
                exit_code = run_serving_sweep.main()

        self.assertEqual(exit_code, 2)
        self.assertEqual(stderr.getvalue(), "error: Start Docker Desktop\n")

    def test_cli_defaults_to_diverse_content_trace(self):
        with patch("sys.argv", ["run_serving_sweep.py"]):
            args = run_serving_sweep.parse_args()

        self.assertEqual(args.trace, "data/trace-round1-diverse-content.jsonl")
        self.assertEqual(args.suite, "experiment1")
        self.assertEqual(args.output_root, "results/experiment1")

    def test_cli_accepts_candidate_preflight_resume_and_force_controls(self):
        with patch(
            "sys.argv",
            [
                "run_serving_sweep.py",
                "--candidate",
                "renderer-2",
                "--preflight-only",
                "--resume",
            ],
        ):
            args = run_serving_sweep.parse_args()

        self.assertEqual(args.candidate, ["renderer-2"])
        self.assertTrue(args.preflight_only)
        self.assertTrue(args.resume)

    def test_experiment1_candidates_change_one_bf16_variable(self):
        candidates = sweep.build_experiment1_candidates()

        self.assertEqual(
            [candidate.name for candidate in candidates],
            [
                "language-only",
                "renderer-2",
                "performance-interactivity",
                "performance-throughput",
                "prefix-off",
            ],
        )
        expected_flags = {
            "language-only": "--language-model-only",
            "renderer-2": "--renderer-num-workers=2",
            "performance-interactivity": "--performance-mode=interactivity",
            "performance-throughput": "--performance-mode=throughput",
            "prefix-off": "--no-enable-prefix-caching",
        }
        for candidate in candidates:
            self.assertIn(expected_flags[candidate.name], candidate.command_args)
            self.assertNotIn("--kv-cache-dtype=fp8", candidate.command_args)
            self.assertNotIn("--max-num-seqs=64", candidate.command_args)

    def test_prefix_off_has_no_positive_prefix_flag(self):
        candidate = sweep.select_candidates(["prefix-off"])[0]

        self.assertIn("--no-enable-prefix-caching", candidate.command_args)
        self.assertNotIn("--enable-prefix-caching", candidate.command_args)

    def test_validate_command_args_rejects_duplicate_flag_keys(self):
        with self.assertRaisesRegex(ValueError, "duplicate vLLM flag"):
            sweep.validate_command_args(("--performance-mode=balanced", "--performance-mode=throughput"))

    def test_select_candidates_rejects_unknown_name(self):
        with self.assertRaisesRegex(ValueError, "unknown Experiment 1 candidate"):
            sweep.select_candidates(["missing"])

    def test_compose_override_replaces_only_model_command(self):
        candidate = sweep.select_candidates(["renderer-2"])[0]
        compose = render_compose_override(candidate)

        self.assertIn("services:", compose)
        self.assertIn("  model:", compose)
        self.assertIn("    command:", compose)
        self.assertIn("- --model=/model #Don't change this to vllm-server", compose)
        self.assertIn("- --served-model-name=Qwen3.5-2B #Don't change this to vllm-server", compose)
        self.assertIn("- --enable-prefix-caching", compose)
        self.assertIn("- --renderer-num-workers=2", compose)
        self.assertNotIn("--kv-cache-dtype=fp8", compose)
        self.assertNotIn("entrypoint:", compose)
        self.assertNotIn("image:", compose)
        self.assertNotIn("volumes:", compose)

    def test_run_command_plan_downs_before_start_then_healthchecks_benchmarks_and_stops(self):
        candidate = sweep.select_candidates(["renderer-2"])[0]
        commands = build_run_commands(
            base_compose=Path("docker-compose.local.yml"),
            override_compose=Path("results/experiment1/renderer-2-01/compose.override.yml"),
            trace=Path("data/trace-round1.jsonl"),
            output_root=Path("results/trace-sweeps"),
            run_id="renderer-2-01",
            python_executable="python",
        )

        self.assertEqual(commands[0], ["docker", "compose", "-f", "docker-compose.local.yml", "-f", "results/experiment1/renderer-2-01/compose.override.yml", "down"])
        self.assertEqual(commands[1], ["docker", "compose", "-f", "docker-compose.local.yml", "-f", "results/experiment1/renderer-2-01/compose.override.yml", "up", "-d", "model"])
        self.assertEqual(
            commands[2],
            [
                "python",
                "scripts/check_server_health.py",
                "--wait",
                "--startup-grace-s",
                "60.0",
                "--poll-interval-s",
                "5.0",
                "--total-timeout-s",
                "300.0",
                "--stable-successes",
                "2",
            ],
        )
        self.assertEqual(
            commands[3],
            [
                "python",
                "scripts/run_trace_benchmark.py",
                "--trace",
                "data/trace-round1.jsonl",
                "--output-root",
                "results/trace-sweeps",
                "--run-id",
                "renderer-2-01",
            ],
        )
        self.assertEqual(commands[4], ["docker", "compose", "-f", "docker-compose.local.yml", "-f", "results/experiment1/renderer-2-01/compose.override.yml", "down"])

    def test_run_commands_invokes_one_stable_health_wait(self):
        commands = [["down"], ["up"], ["health"], ["benchmark"], ["cleanup"]]

        with patch.object(sweep.subprocess, "run") as run:
            sweep.run_commands(commands)

        self.assertEqual(
            run.call_args_list,
            [
                call(commands[0], check=False),
                call(commands[1], check=True),
                call(commands[2], check=True),
                call(commands[3], check=True),
                call(commands[4], check=False),
            ],
        )


if __name__ == "__main__":
    unittest.main()
