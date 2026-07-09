from __future__ import annotations

import argparse
import json
from pathlib import Path

from inference_opt.benchmark.scoring import accuracy_multiplier
from inference_opt.eval.gpqa import LOCAL_BASELINE_ACCURACY, score_gpqa_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score GPQA answers from a trace benchmark requests.jsonl file.")
    parser.add_argument("--answer-key", default="data/traces/gpqa-diamond-120.answers.json")
    parser.add_argument("--requests", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--baseline-accuracy", type=float, default=LOCAL_BASELINE_ACCURACY)
    return parser.parse_args()


def _load_answer_key(path: Path) -> dict[int, str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {int(item["request_id"]): str(item["correct_label"]) for item in raw["answers"]}


def _load_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def build_report(
    answer_key: dict[int, str],
    rows: list[dict],
    *,
    baseline_accuracy: float = LOCAL_BASELINE_ACCURACY,
) -> dict:
    report = score_gpqa_results(answer_key, rows)
    accuracy = float(report["accuracy"])
    report["baseline_accuracy"] = baseline_accuracy
    report["accuracy_delta"] = baseline_accuracy - accuracy
    report["accuracy_multiplier"] = accuracy_multiplier(
        accuracy,
        baseline_accuracy=baseline_accuracy,
    )
    return report


def main() -> int:
    args = parse_args()
    requests_path = Path(args.requests)
    output_path = Path(args.output) if args.output else requests_path.parent / "gpqa_accuracy.json"
    report = build_report(
        _load_answer_key(Path(args.answer_key)),
        _load_rows(requests_path),
        baseline_accuracy=args.baseline_accuracy,
    )
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output_path.as_posix())
    print(f"accuracy={report['accuracy']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
