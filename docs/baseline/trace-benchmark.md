# Trace Benchmark

This benchmark replays a JSONL trace against an OpenAI-compatible vLLM server and computes the ERS proxy from `AGENTS.md`.

## Inputs

- Primary trace: `data/trace-round1-diverse-content.jsonl`
- Legacy trace: `data/trace-round1.jsonl`
- Server: `http://localhost:8000/v1`
- Endpoint: `POST /chat/completions`
- Output: `results/baseline/<run-id>/summary.json` and `requests.jsonl`

## Scoring

The benchmark uses:

- `F_ttft = 100 ms`
- `C_ttft = 1500 ms`
- `F_tpot = 20 ms`
- `C_tpot = 45 ms`
- `gamma = 2`
- `w = 0.5`

Requests with an error, timeout, or zero output tokens score `0`.

## Measurement Contract

- TTFT is measured from request send start to the first non-empty streamed content or reasoning chunk.
- Completion tokens come from OpenAI-compatible streaming `usage.completion_tokens` when available.
- Prompt and completion token usage are mandatory; a missing usage block fails the request instead of falling back to SSE chunk count.
- TPOT is measured as `(total_latency_ms - TTFT_ms) / (completion_tokens - 1)` for multi-token responses.
- The runner requests `stream_options.include_usage = true`, reuses one HTTP client, and permits one active connection per trace record.
- Request rows include scheduled offset, dispatch offset, dispatch lag, and exact prompt-token usage.
- Summaries include `measurement_version`, trace SHA-256, makespan, dispatch-lag P95, ERC, and prompt-token percentiles.
- Summary output keeps the original local fields and also reports BTC-like fields such as `passed_slo`, `failed_count`, `ttft_p50_ms`, `ttft_p95_ms`, `tbt_median_ms`, `ers`, and `final_score`.

## Run

Preferred H1 cold protocol:

```powershell
python scripts/run_serving_sweep.py --mode baseline --repeat 3 --output-root results\trace-baseline-h01
```

Each repeat runs `docker-compose down`, starts a fresh model container, waits
quietly for 60 seconds, then requires two consecutive successful `/v1/models`
checks before replaying the trace and stopping the container. No generation
warmup request is sent.

After the repeats finish, summarize the cold run group:

```powershell
python scripts/summarize_trace_runs.py --root results\trace-baseline-h01 --min-runs 3 --expected-total-count 120
```

Use `ready_for_comparison = true` as the local gate before comparing
optimization candidates.

## Required Checks

These checks are mandatory before accepting a local baseline or candidate:

- `total_count = 120` for every run, because the score averages across all 120 trace requests.
- `failed_count = 0` for every run, because errors, timeouts, and zero-token responses score `0`.
- Every successful request has prompt and completion usage; missing usage is a measurement failure.
- `dispatch_lag_p95_ms < 25` so the harness releases requests before the next 25 ms trace arrival.
- At least 3 independent cold repeats for screening, because one run is too noisy and warm-state contamination is easy to miss.
- Compare median `ttft_p50_ms`, `ttft_p95_ms`, `tbt_median_ms`, and `ers`; `tbt_median_ms` is especially important because AGENTS.md gives TPOT a narrow 20-45 ms scoring window.
- Run GPQA for any numerical or correctness-risk change such as KV FP8, prefix/chunked-prefill behavior, quantization, speculative decoding, or framework changes, because the accuracy multiplier can zero the final score.

These checks are useful but can be skipped for early local screening:

- 5 cold repeats instead of 3. Use this for finalists, not every small sweep.
- GPQA for pure scheduler-only flags such as `--max-num-seqs` or `--max-num-batched-tokens`, if the model output path is unchanged.
- Official leaderboard submission for every local candidate. Submit only candidates that beat the H2 cold baseline by more than run-to-run noise.
- Exact match to BTC hidden score. Local L4 measurements are a relative filter, not a conversion to H200 MIG score.

Start the local baseline server first:

```powershell
docker-compose -f docker-compose.local.yml up model
```

For manual debugging against an already-running server, run:

```powershell
python scripts/check_server_health.py
python scripts/run_trace_benchmark.py --trace data\trace-round1-diverse-content.jsonl --output-root results\trace-baseline-h01 --run-id baseline-01
python scripts/run_trace_benchmark.py --trace data\trace-round1-diverse-content.jsonl --output-root results\trace-baseline-h01 --run-id baseline-02
python scripts/run_trace_benchmark.py --trace data\trace-round1-diverse-content.jsonl --output-root results\trace-baseline-h01 --run-id baseline-03
```

For a quick functional check that does not wait for trace timestamps:

```powershell
python scripts/run_trace_benchmark.py --no-respect-timestamps --run-id smoke
```

This local score is a proxy. It is not expected to match the leaderboard unless the hardware is comparable to the organizer `1g.18gb` H200 MIG environment.
On L4, use the three-run median raw TTFT, TPOT, makespan, and output rate
recorded in `docs/baseline/results.md`; official ERS is saturated near zero.

The primary trace uses coherent content but remains prefix-heavy: 20 sessions
share one long system prompt and each session grows across six turns. Prefix
cache conclusions are therefore workload-specific. Do not mix legacy H0
artifacts with H0.1 results; the client concurrency and token-measurement
contracts changed.

For GPQA-derived local quality checks, see `docs/baseline/gpqa-benchmark.md`.
