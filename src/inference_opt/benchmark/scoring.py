from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ScoreConfig:
    ttft_floor_ms: float = 100.0
    ttft_ceiling_ms: float = 1500.0
    tpot_floor_ms: float = 20.0
    tpot_ceiling_ms: float = 45.0
    gamma: float = 2.0
    ttft_weight: float = 0.5
    baseline_accuracy: float = 0.40


@dataclass(frozen=True)
class RequestMeasurement:
    ttft_ms: float | None = None
    tpot_ms: float | None = None
    prompt_tokens: int | None = None
    output_tokens: int = 0
    error: str | None = None
    dispatch_lag_ms: float | None = None


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _term(value: float, floor: float, ceiling: float, gamma: float) -> float:
    return _clamp((ceiling - value) / (ceiling - floor), 0.0, 1.0) ** gamma


def request_score(measurement: RequestMeasurement, config: ScoreConfig) -> float:
    if measurement.error or measurement.output_tokens <= 0:
        return 0.0
    if measurement.ttft_ms is None or measurement.tpot_ms is None:
        return 0.0

    s_ttft = _term(measurement.ttft_ms, config.ttft_floor_ms, config.ttft_ceiling_ms, config.gamma)
    s_tpot = _term(measurement.tpot_ms, config.tpot_floor_ms, config.tpot_ceiling_ms, config.gamma)
    return config.ttft_weight * s_ttft + (1.0 - config.ttft_weight) * s_tpot


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    index = max(0, min(len(sorted_values) - 1, math.ceil(q * len(sorted_values)) - 1))
    return sorted_values[index]


def summarize_scores(measurements: list[RequestMeasurement], config: ScoreConfig) -> dict[str, float | int | None]:
    scores = [request_score(measurement, config) for measurement in measurements]
    success_count = sum(1 for measurement in measurements if not measurement.error and measurement.output_tokens > 0)
    error_count = len(measurements) - success_count
    passed_slo = sum(1 for score in scores if score > 0.0)
    erc = passed_slo / len(measurements) if measurements else 0.0
    effective_request_score = sum(scores) / len(scores) if scores else 0.0
    ttft_values = [measurement.ttft_ms for measurement in measurements if measurement.ttft_ms is not None]
    tpot_values = [measurement.tpot_ms for measurement in measurements if measurement.tpot_ms is not None]
    prompt_token_values = [
        float(measurement.prompt_tokens)
        for measurement in measurements
        if measurement.prompt_tokens is not None
    ]
    dispatch_lag_values = [
        measurement.dispatch_lag_ms
        for measurement in measurements
        if measurement.dispatch_lag_ms is not None
    ]
    score_100x_ers = 100.0 * effective_request_score
    return {
        "request_count": len(measurements),
        "total_count": len(measurements),
        "success_count": success_count,
        "error_count": error_count,
        "failed_count": error_count,
        "passed_slo": passed_slo,
        "erc": erc,
        "ttft_p50_ms": _percentile(ttft_values, 0.50),
        "ttft_p95_ms": _percentile(ttft_values, 0.95),
        "tbt_median_ms": _percentile(tpot_values, 0.50),
        "prompt_tokens_p50": _percentile(prompt_token_values, 0.50),
        "prompt_tokens_p95": _percentile(prompt_token_values, 0.95),
        "dispatch_lag_p95_ms": _percentile(dispatch_lag_values, 0.95),
        "effective_request_score": effective_request_score,
        "score_100x_ers": score_100x_ers,
        "ers": score_100x_ers,
        "final_score": score_100x_ers,
    }


def accuracy_multiplier(accuracy: float, *, baseline_accuracy: float = 0.40) -> float:
    epsilon = 1e-12
    delta = baseline_accuracy - accuracy
    if delta <= 0.10 + epsilon:
        return 1.0
    if delta >= 0.16 - epsilon:
        return 0.0
    return 1.0 - (delta - 0.10) / 0.06


def final_score(
    *,
    effective_request_score: float,
    accuracy: float,
    baseline_accuracy: float = 0.40,
) -> float:
    return 100.0 * effective_request_score * accuracy_multiplier(
        accuracy,
        baseline_accuracy=baseline_accuracy,
    )
