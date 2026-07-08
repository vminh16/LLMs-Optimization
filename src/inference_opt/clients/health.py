from __future__ import annotations

from typing import Any

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
