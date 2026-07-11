from pathlib import Path
import unittest
from unittest.mock import call, patch

from inference_opt.serving import sweep
from inference_opt.serving.sweep import (
    build_kv_fp8_batched_token_candidates,
    build_kv_fp8_seq_candidates,
    build_run_commands,
    render_compose_override,
)
from scripts import run_serving_sweep


class ServingSweepTest(unittest.TestCase):
    def test_cli_defaults_to_diverse_content_trace(self):
        with patch("sys.argv", ["run_serving_sweep.py"]):
            args = run_serving_sweep.parse_args()

        self.assertEqual(args.trace, "data/trace-round1-diverse-content.jsonl")

    def test_baseline_candidate_keeps_only_baseline_args(self):
        candidates = sweep.build_baseline_candidates()

        self.assertEqual([candidate.name for candidate in candidates], ["baseline-cold"])
        self.assertEqual(candidates[0].extra_args, ())
        self.assertIn("--enable-prefix-caching", candidates[0].command_args)
        self.assertNotIn("--kv-cache-dtype=fp8", candidates[0].command_args)
        self.assertNotIn("--max-num-seqs=64", candidates[0].command_args)

    def test_seq_sweep_candidates_keep_kv_fp8_and_change_one_scheduler_arg(self):
        candidates = build_kv_fp8_seq_candidates([32, 64])

        self.assertEqual([candidate.name for candidate in candidates], ["kv-fp8-seqs-32", "kv-fp8-seqs-64"])
        for candidate, value in zip(candidates, [32, 64]):
            self.assertIn("--kv-cache-dtype=fp8", candidate.extra_args)
            self.assertIn("--calculate-kv-scales", candidate.extra_args)
            self.assertIn(f"--max-num-seqs={value}", candidate.extra_args)
            self.assertNotIn("--prefix-caching-hash-algo=xxhash", candidate.extra_args)
            self.assertNotIn("--quantization=fp8", candidate.extra_args)

    def test_batched_token_candidates_pin_best_sequence_limit(self):
        candidates = build_kv_fp8_batched_token_candidates([2048, 4096], max_num_seqs=64)

        self.assertEqual(
            [candidate.name for candidate in candidates],
            ["kv-fp8-seqs-64-btokens-2048", "kv-fp8-seqs-64-btokens-4096"],
        )
        for candidate, value in zip(candidates, [2048, 4096]):
            self.assertIn("--max-num-seqs=64", candidate.extra_args)
            self.assertIn(f"--max-num-batched-tokens={value}", candidate.extra_args)

    def test_compose_override_replaces_only_model_command(self):
        candidate = build_kv_fp8_seq_candidates([64])[0]
        compose = render_compose_override(candidate)

        self.assertIn("services:", compose)
        self.assertIn("  model:", compose)
        self.assertIn("    command:", compose)
        self.assertIn("- --model=/model #Don't change this to vllm-server", compose)
        self.assertIn("- --served-model-name=Qwen3.5-2B #Don't change this to vllm-server", compose)
        self.assertIn("- --enable-prefix-caching", compose)
        self.assertIn("- --kv-cache-dtype=fp8", compose)
        self.assertIn("- --max-num-seqs=64", compose)
        self.assertNotIn("entrypoint:", compose)
        self.assertNotIn("image:", compose)
        self.assertNotIn("volumes:", compose)

    def test_run_command_plan_downs_before_start_then_healthchecks_benchmarks_and_stops(self):
        candidate = build_kv_fp8_seq_candidates([64])[0]
        commands = build_run_commands(
            base_compose=Path("docker-compose.local.yml"),
            override_compose=Path("results/sweeps/_overrides/kv-fp8-seqs-64/compose.override.yml"),
            trace=Path("data/trace-round1.jsonl"),
            output_root=Path("results/trace-sweeps"),
            run_id="kv-fp8-seqs-64-01",
            python_executable="python",
        )

        self.assertEqual(commands[0], ["docker-compose", "-f", "docker-compose.local.yml", "-f", "results/sweeps/_overrides/kv-fp8-seqs-64/compose.override.yml", "down"])
        self.assertEqual(commands[1], ["docker-compose", "-f", "docker-compose.local.yml", "-f", "results/sweeps/_overrides/kv-fp8-seqs-64/compose.override.yml", "up", "-d", "model"])
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
                "kv-fp8-seqs-64-01",
            ],
        )
        self.assertEqual(commands[4], ["docker-compose", "-f", "docker-compose.local.yml", "-f", "results/sweeps/_overrides/kv-fp8-seqs-64/compose.override.yml", "down"])

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
