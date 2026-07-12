# Performance Harness

## Active Contract

- Trace: `data/trace-round1-diverse-content.jsonl`, 120 requests.
- Streaming usage is mandatory; SSE chunk count is never treated as token count.
- `TPOT = (total latency - TTFT) / (completion tokens - 1)`.
- The client permits 120 active HTTP connections and preserves trace arrival times.
- Every run records trace SHA-256, `measurement_version=h0.1`, makespan,
  dispatch lag, prompt tokens, TTFT, TBT, ERS, and request-level errors.
- Readiness uses a quiet grace period followed by two successful `/v1/models`
  checks. No generation warmup is sent.

## Acceptance Gate

A comparable run requires:

- `total_count = 120`;
- `failed_count = 0`;
- the same trace SHA-256 and measurement version;
- `dispatch_lag_p95_ms < 25`;
- three cold runs for a finalist.

Use TBT first, then makespan and TTFT P95. GPQA remains a separate harness and
is mandatory for precision, cache-correctness, speculative decoding, or
framework changes.

## Commands

Run a trace against an already-running model:

```bash
python scripts/run_trace_benchmark.py \
  --trace data/trace-round1-diverse-content.jsonl \
  --output-root results/manual \
  --run-id candidate-01
```

Summarize repeated standalone runs:

```bash
python scripts/summarize_trace_runs.py \
  --root results/manual \
  --min-runs 3 \
  --expected-total-count 120
```

For Experiment 1 lifecycle, preflight, cleanup, manifests, and comparison, use
the commands in `docs/optimization/free-wins.md`.

The L4 harness is a relative filter. It does not predict the absolute score on
the organizer `1g.18gb` H200 MIG instance.
