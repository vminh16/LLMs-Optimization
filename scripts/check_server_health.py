from __future__ import annotations

import argparse
import json

from inference_opt.clients.health import fetch_models, require_model, wait_for_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check the local vLLM OpenAI-compatible server.")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--expected-model", default="Qwen3.5-2B")
    parser.add_argument("--timeout-s", type=float, default=5.0)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--startup-grace-s", type=float, default=60.0)
    parser.add_argument("--poll-interval-s", type=float, default=5.0)
    parser.add_argument("--total-timeout-s", type=float, default=300.0)
    parser.add_argument("--stable-successes", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.wait:
        wait_for_model(
            args.base_url,
            args.expected_model,
            request_timeout_s=args.timeout_s,
            startup_grace_s=args.startup_grace_s,
            poll_interval_s=args.poll_interval_s,
            total_timeout_s=args.total_timeout_s,
            stable_successes=args.stable_successes,
        )
    else:
        payload = fetch_models(args.base_url, args.timeout_s)
        require_model(payload, args.expected_model)
    print(json.dumps({"ok": True, "expected_model": args.expected_model}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
