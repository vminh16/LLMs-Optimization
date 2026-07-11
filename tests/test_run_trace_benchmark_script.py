import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import AsyncMock, patch

from inference_opt.benchmark.runner import BenchmarkResult
from scripts import run_trace_benchmark


class RunTraceBenchmarkScriptTest(unittest.TestCase):
    def test_parse_args_defaults_to_diverse_content_trace(self):
        with patch("sys.argv", ["run_trace_benchmark.py"]):
            args = run_trace_benchmark.parse_args()

        self.assertEqual(args.trace, "data/trace-round1-diverse-content.jsonl")

    def test_main_records_trace_identity(self):
        with TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.jsonl"
            trace_path.write_text("{}\n", encoding="utf-8")
            expected_hash = hashlib.sha256(trace_path.read_bytes()).hexdigest()
            result = BenchmarkResult(requests=[], summary={"ers": 1.0})
            written_results = []

            with (
                patch(
                    "sys.argv",
                    [
                        "run_trace_benchmark.py",
                        "--trace",
                        str(trace_path),
                        "--output-root",
                        tmp,
                        "--run-id",
                        "test-run",
                    ],
                ),
                patch.object(run_trace_benchmark, "load_trace", return_value=[]),
                patch.object(run_trace_benchmark, "run_benchmark", new=AsyncMock(return_value=result)),
                patch.object(
                    run_trace_benchmark,
                    "write_benchmark_result",
                    side_effect=lambda value, _output_dir: written_results.append(value),
                ),
            ):
                exit_code = run_trace_benchmark.main()

        written_summary = written_results[0].summary
        self.assertEqual(exit_code, 0)
        self.assertEqual(written_summary["trace_path"], trace_path.as_posix())
        self.assertEqual(
            written_summary["trace_sha256"],
            expected_hash,
        )


if __name__ == "__main__":
    unittest.main()
