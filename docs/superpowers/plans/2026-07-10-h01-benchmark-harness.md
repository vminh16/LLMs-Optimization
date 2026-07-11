# H0.1 Benchmark Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the local L4 benchmark replay the 120-request trace without a client-side queue, require exact token usage, expose timing drift, and wait quietly for stable server readiness.

**Architecture:** Keep request measurement in `benchmark/runner.py`, aggregate BTC-like fields in `benchmark/scoring.py`, readiness polling in `clients/health.py`, and Docker command orchestration in `serving/sweep.py`. CLI scripts only parse arguments and attach trace identity. The scoring formula and serving configuration remain unchanged.

**Tech Stack:** Python 3.11+, asyncio, httpx 0.28, unittest/pytest, Docker Compose.

## Global Constraints

- Preserve `python3 -m vllm.entrypoints.openai.api_server` exactly.
- Preserve all ERS constants and the accuracy gate from `AGENTS.md`.
- Use `data/trace-round1-diverse-content.jsonl` for new local performance runs.
- Do not add a generation warmup request.
- Do not add dependencies or new source-depth levels.
- Do not rewrite existing result artifacts.

---

### Task 1: Exact Usage And HTTP Capacity

**Files:**
- Modify: `src/inference_opt/benchmark/runner.py`
- Modify: `src/inference_opt/benchmark/scoring.py`
- Test: `tests/test_benchmark_runner.py`

**Interfaces:**
- Consumes: OpenAI streaming usage payloads with `prompt_tokens` and `completion_tokens`.
- Produces: `RequestMeasurement.prompt_tokens`, exact `output_tokens`, and an `AsyncClient` whose maximum active connections equal the trace record count.

- [ ] **Step 1: Write failing tests for exact usage and missing usage**

Add tests that stream a final usage chunk, assert `prompt_tokens` and TPOT, and assert a stream without usage raises `ValueError`:

```python
self.assertEqual(result["prompt_tokens"], 17)
self.assertEqual(result["output_tokens"], 5)
self.assertAlmostEqual(result["tpot_ms"], 250.0)

with self.assertRaisesRegex(ValueError, "usage"):
    asyncio.run(run_without_usage())
```

- [ ] **Step 2: Run the focused tests and confirm RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_benchmark_runner.py
```

Expected: failures because `prompt_tokens` is absent and missing usage still falls back to chunk count.

- [ ] **Step 3: Implement strict usage parsing**

Replace the completion-only helper with:

```python
def _usage_from_chunk(payload: dict[str, Any]) -> tuple[int | None, int | None]:
    usage = payload.get("usage") or {}
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    return (
        int(prompt_tokens) if prompt_tokens is not None else None,
        int(completion_tokens) if completion_tokens is not None else None,
    )
```

Track both values, require them after the stream, and return `prompt_tokens` with the existing fields. Add `prompt_tokens: int | None = None` to `RequestMeasurement`.

- [ ] **Step 4: Write a failing test for 120 active client connections**

Patch the runner's `httpx.AsyncClient` and `send_openai_streaming_request`, run 120 records, then assert:

```python
self.assertEqual(seen_limits.max_connections, 120)
```

- [ ] **Step 5: Run the capacity test and confirm RED**

Expected: `AsyncClient` uses the httpx default instead of an explicit 120-request limit.

- [ ] **Step 6: Implement bounded trace-sized capacity**

Construct the shared client with:

```python
connection_limit = max(1, len(records))
limits = httpx.Limits(
    max_connections=connection_limit,
    max_keepalive_connections=min(20, connection_limit),
)
```

- [ ] **Step 7: Run Task 1 tests and confirm GREEN**

Run the focused runner and scoring tests. Expected: all pass.

---

### Task 2: Scheduling Telemetry And Trace Identity

**Files:**
- Modify: `src/inference_opt/benchmark/runner.py`
- Modify: `src/inference_opt/benchmark/scoring.py`
- Modify: `src/inference_opt/benchmark/report.py`
- Modify: `scripts/run_trace_benchmark.py`
- Test: `tests/test_benchmark_runner.py`
- Test: `tests/test_benchmark_scoring.py`
- Test: `tests/test_benchmark_report.py`

**Interfaces:**
- Consumes: one monotonic trace start time and each record's timestamp.
- Produces: row-level schedule fields plus `erc`, `makespan_ms`, prompt-token percentiles, dispatch-lag P95, trace path/hash, and `measurement_version=h0.1`.

- [ ] **Step 1: Write failing aggregation tests**

Construct measurements with prompt-token and dispatch-lag values and assert:

```python
self.assertEqual(summary["erc"], 0.5)
self.assertEqual(summary["prompt_tokens_p50"], 100)
self.assertEqual(summary["prompt_tokens_p95"], 200)
self.assertEqual(summary["dispatch_lag_p95_ms"], 7.0)
```

- [ ] **Step 2: Confirm aggregation tests fail**

Run `tests/test_benchmark_scoring.py`; expected keys are absent.

- [ ] **Step 3: Add measurement fields and aggregates**

Extend `RequestMeasurement` with `dispatch_lag_ms`. In `summarize_scores`, aggregate non-null values with the existing nearest-rank percentile helper and compute:

```python
erc = passed_slo / len(measurements) if measurements else 0.0
```

- [ ] **Step 4: Write a failing runner timing test**

Use two fake records and a patched monotonic clock to assert each row has `scheduled_offset_ms`, `dispatch_offset_ms`, and `dispatch_lag_ms`, while the summary has `makespan_ms` and `measurement_version`.

- [ ] **Step 5: Confirm timing test fails**

Run `tests/test_benchmark_runner.py`; expected telemetry keys are absent.

- [ ] **Step 6: Implement shared-clock telemetry**

Capture `run_started = perf_counter()` once in `run_benchmark`, pass it and the scheduled offset into `_run_one`, and add:

```python
dispatch_offset_ms = (perf_counter() - run_started) * 1000.0
dispatch_lag_ms = max(0.0, dispatch_offset_ms - scheduled_offset_ms)
```

After `gather`, add elapsed makespan and `measurement_version="h0.1"` to the summary.

- [ ] **Step 7: Add trace identity in the CLI**

Before writing results, copy the summary and add:

```python
summary["trace_path"] = trace_path.as_posix()
summary["trace_sha256"] = hashlib.sha256(trace_path.read_bytes()).hexdigest()
```

- [ ] **Step 8: Extend repeated-run reporting**

Add `erc`, `makespan_ms`, `dispatch_lag_p95_ms`, `prompt_tokens_p50`, and `prompt_tokens_p95` to `METRIC_KEYS`, then update report tests.

- [ ] **Step 9: Run Task 2 tests and confirm GREEN**

Run runner, scoring, and report tests. Expected: all pass.

---

### Task 3: Quiet Stable Readiness

**Files:**
- Modify: `src/inference_opt/clients/health.py`
- Modify: `scripts/check_server_health.py`
- Modify: `src/inference_opt/serving/sweep.py`
- Modify: `scripts/run_serving_sweep.py`
- Test: `tests/test_phase0_setup.py`
- Test: `tests/test_serving_sweep.py`

**Interfaces:**
- Consumes: base URL, expected model, startup grace, poll interval, total timeout, and required consecutive successes.
- Produces: `wait_for_model(...) -> dict[str, Any]` and one health subprocess per cold run.

- [ ] **Step 1: Write failing readiness tests**

Patch `fetch_models`, `time.sleep`, and `time.monotonic` so two connection failures followed by two valid model responses succeed without printing. Add a timeout test that asserts one `TimeoutError` contains the last failure.

- [ ] **Step 2: Confirm readiness tests fail**

Run the focused health/setup tests; expected `wait_for_model` is undefined.

- [ ] **Step 3: Implement `wait_for_model`**

Add a polling loop that sleeps through the grace period, catches `httpx.HTTPError` and `ValueError`, resets the consecutive-success counter after a failure, and returns only after the configured number of successes.

- [ ] **Step 4: Extend the health CLI**

Add `--wait`, `--startup-grace-s=60`, `--poll-interval-s=5`, `--total-timeout-s=300`, and `--stable-successes=2`. Keep the current one-shot behavior when `--wait` is absent.

- [ ] **Step 5: Write failing sweep command tests**

Assert the generated health command contains the wait flags and that `run_commands` invokes it exactly once before the benchmark.

- [ ] **Step 6: Confirm sweep tests fail**

Run `tests/test_serving_sweep.py`; expected command shape and invocation count differ.

- [ ] **Step 7: Simplify sweep readiness orchestration**

Build one `check_server_health.py --wait ...` command and replace the outer retry loop with one checked subprocess call. Preserve `docker-compose down` in `finally`.

- [ ] **Step 8: Run Task 3 tests and confirm GREEN**

Run health/setup and sweep tests. Expected: all pass with no retry tracebacks.

---

### Task 4: Primary Trace And Documentation

**Files:**
- Modify: `scripts/run_trace_benchmark.py`
- Modify: `scripts/run_serving_sweep.py`
- Modify: `docs/baseline/trace-benchmark.md`
- Modify: `docs/baseline/results.md`
- Modify: `docs/optimization/free-wins.md`
- Test: `tests/test_phase0_setup.py`

**Interfaces:**
- Consumes: `data/trace-round1-diverse-content.jsonl`.
- Produces: new-run defaults and commands that consistently identify the primary trace while retaining the old file unchanged.

- [ ] **Step 1: Write a failing default-trace test**

Assert both benchmark and sweep CLIs default to `data/trace-round1-diverse-content.jsonl`.

- [ ] **Step 2: Confirm the test fails**

Expected: both CLIs still use `data/trace-round1.jsonl`.

- [ ] **Step 3: Change defaults and update documentation**

Document the new trace as primary, the old trace as legacy, the prefix-heavy limitation, H0.1 fields, quiet readiness, and the required three cold baseline runs. State that pre-H0.1 artifacts are not numerically comparable.

- [ ] **Step 4: Run documentation/config tests and confirm GREEN**

Run `tests/test_phase0_setup.py` and relevant CLI tests. Expected: all pass.

---

### Task 5: Verification And Handoff

**Files:**
- Verify all modified files.
- Do not modify Docker submission configuration.

**Interfaces:**
- Consumes: completed H0.1 implementation.
- Produces: a tested harness and exact user commands for locking the new baseline.

- [ ] **Step 1: Run focused H0.1 tests**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_benchmark_runner.py tests\test_benchmark_scoring.py tests\test_benchmark_report.py tests\test_serving_sweep.py tests\test_phase0_setup.py
```

Expected: all focused tests pass.

- [ ] **Step 2: Run the full test suite**

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Expected: all tests pass. If the known compose expectation remains stale, report it separately and do not alter serving flags to satisfy it.

- [ ] **Step 3: Preview the cold baseline command**

```powershell
.\.venv\Scripts\python.exe scripts\run_serving_sweep.py --mode baseline --repeat 3 --output-root results\trace-baseline-h01 --dry-run
```

Expected: each repeat performs `down`, `up`, one stable wait command, the new trace benchmark, and `down`.

- [ ] **Step 4: Provide the real baseline command**

```powershell
.\.venv\Scripts\python.exe scripts\run_serving_sweep.py --mode baseline --repeat 3 --output-root results\trace-baseline-h01
```

Do not run the three GPU baselines automatically unless the user asks; they consume substantial local time and GPU resources.
