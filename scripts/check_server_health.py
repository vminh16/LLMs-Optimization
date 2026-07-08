from __future__ import annotations

import argparse
import json

from inference_opt.clients.health import fetch_models, require_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check the local vLLM OpenAI-compatible server.")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--expected-model", default="Qwen3.5-2B")
    parser.add_argument("--timeout-s", type=float, default=5.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = fetch_models(args.base_url, args.timeout_s)
    require_model(payload, args.expected_model)
    print(json.dumps({"ok": True, "expected_model": args.expected_model}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
