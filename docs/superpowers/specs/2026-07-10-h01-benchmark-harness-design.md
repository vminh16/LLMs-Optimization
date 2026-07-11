# H0.1 Benchmark Harness Design

## Goal

Make the local L4 benchmark a reproducible relative-ranking tool before
running independent serving experiments. The benchmark must replay all 120
requests without an unintended client-side concurrency limit, use exact vLLM
token usage, expose scheduling drift, and start only after the server is
stably ready.

The local score remains a proxy. It is not expected to reproduce the absolute
score from the organizer H200 MIG environment.

## Trace Decision

`data/trace-round1-diverse-content.jsonl` is the primary local performance
trace for new experiments. It preserves the old trace's request count,
arrival pattern, context-length envelope, deterministic decoding settings, and
20-session/six-turn structure while replacing synthetic word sequences with
coherent long-context conversations.

`data/trace-round1.jsonl` remains unchanged as a legacy reference. Results
from the two traces must not be mixed in one candidate comparison. Both traces
are prefix-heavy, so prefix-cache results are workload-specific and require an
organizer submission before promotion.

## Measurement Changes

### HTTP concurrency

The shared `httpx.AsyncClient` must permit at least one active connection per
trace record. For the competition trace this means 120 active connections.
This removes the current default limit of 100, which queues the final 20
requests inside the benchmark client instead of at vLLM.

### Token usage

Every successful streamed response must include OpenAI-compatible usage data.
The runner records both `prompt_tokens` and `completion_tokens`. Missing
completion usage is a measurement failure; the runner must not substitute the
number of SSE chunks because a chunk can contain multiple tokens.

TPOT remains:

```text
(total_latency_ms - TTFT_ms) / (completion_tokens - 1)
```

The ERS formula, constants, and accuracy gate from `AGENTS.md` do not change.

### Scheduling telemetry

Each request row records:

- `scheduled_offset_ms`: intended release time relative to trace start.
- `dispatch_offset_ms`: actual time the request coroutine begins sending.
- `dispatch_lag_ms`: actual offset minus scheduled offset.
- `prompt_tokens`: prompt usage reported by vLLM.

The summary adds:

- `erc`: `passed_slo / total_count`.
- `makespan_ms`: wall-clock duration from trace start through the final
  response.
- `dispatch_lag_p95_ms`.
- `prompt_tokens_p50` and `prompt_tokens_p95`.
- `measurement_version`: fixed value `h0.1`.

The benchmark CLI also records the trace path and SHA-256 in the summary so
results produced from different traces cannot be mistaken for one another.

## Readiness Protocol

Docker Compose starts one cold model container per run. The runner then:

1. Waits for a configurable quiet startup grace period, default 60 seconds.
2. Polls `/v1/models` every 5 seconds without printing expected connection
   errors.
3. Requires two consecutive successful checks for `Qwen3.5-2B`.
4. Fails after a five-minute total readiness timeout with one concise error.
5. Starts the trace without a generation warmup request.

This protocol avoids noisy early health failures while preserving the
organizer-like `warmup_count=0` behavior.

## Module Boundaries

- `src/inference_opt/benchmark/runner.py`: streaming measurement, client
  capacity, scheduling telemetry, and run-level makespan.
- `src/inference_opt/benchmark/scoring.py`: BTC-like aggregate fields only;
  the scoring formula remains unchanged.
- `src/inference_opt/clients/health.py`: reusable stable-readiness polling.
- `src/inference_opt/serving/sweep.py`: command orchestration only; delegates
  readiness to the health CLI.
- `scripts/`: argument parsing and thin calls into `src`.
- `tests/`: behavior tests for every new contract.

No new utility module or dependency is introduced.

## Error Handling

- Missing completion-token usage fails only that request and gives it score
  zero, matching the existing request-error contract.
- Readiness ignores expected connection and HTTP failures until timeout.
- A failed benchmark still triggers `docker-compose down` through the existing
  `finally` block.
- Existing result files are not rewritten or migrated.

## Verification And Acceptance

Automated verification must prove:

- the benchmark client capacity is at least the number of trace records;
- missing usage cannot silently fall back to chunk count;
- prompt/completion tokens and TPOT are recorded correctly;
- dispatch lag and makespan use one shared monotonic run clock;
- readiness waits quietly and requires consecutive successes;
- all existing scoring tests retain their current results.

After automated tests pass, lock a new three-run cold baseline on the new
trace. Each run must have 120 requests, zero failures, complete usage data,
and dispatch-lag P95 below 25 ms. Across the three runs, target TBT-median
spread below 1% and makespan spread below 3% before starting Experiment 1.

## Non-Goals

- Do not change vLLM serving flags or the submission compose file.
- Do not add a generation warmup.
- Do not rewrite or randomize either trace.
- Do not recalibrate the organizer scoring constants.
- Do not claim that local L4 ranking guarantees H200 MIG ranking.
