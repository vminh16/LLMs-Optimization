from __future__ import annotations

from typing import Any
import time

import httpx


def build_models_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/models"


def extract_model_ids(payload: dict[str, Any]) -> list[str]:
    return [str(item.get("id")) for item in payload.get("data", []) if item.get("id")]


def require_model(payload: dict[str, Any], expected_model: str) -> None:
    model_ids = extract_model_ids(payload)
    if expected_model not in model_ids:
        raise ValueError(f"Expected model {expected_model!r}; available models: {model_ids}")


def fetch_models(base_url: str, timeout_s: float) -> dict[str, Any]:
    response = httpx.get(build_models_url(base_url), timeout=timeout_s)
    response.raise_for_status()
    return response.json()


def wait_for_model(
    base_url: str,
    expected_model: str,
    *,
    request_timeout_s: float,
    startup_grace_s: float,
    poll_interval_s: float,
    total_timeout_s: float,
    stable_successes: int,
) -> dict[str, Any]:
    if stable_successes < 1:
        raise ValueError("stable_successes must be at least 1")

    if startup_grace_s > 0:
        time.sleep(startup_grace_s)
    started = time.monotonic()

    consecutive_successes = 0
    last_error: Exception | None = None
    while True:
        try:
            payload = fetch_models(base_url, request_timeout_s)
            require_model(payload, expected_model)
        except (httpx.HTTPError, ValueError) as exc:
            last_error = exc
            consecutive_successes = 0
        else:
            consecutive_successes += 1
            if consecutive_successes >= stable_successes:
                return payload

        elapsed_s = time.monotonic() - started
        if elapsed_s >= total_timeout_s:
            detail = str(last_error) if last_error else "server did not become stable"
            raise TimeoutError(f"Model readiness timed out: {detail}") from last_error
        time.sleep(min(poll_interval_s, total_timeout_s - elapsed_s))
