# Trace Benchmark

This benchmark replays a JSONL trace against an OpenAI-compatible vLLM server and computes the ERS proxy from `AGENTS.md`.

## Inputs

- Trace: `data/trace-round1.jsonl`
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

## Run

Start the local baseline server first:

```powershell
docker-compose -f docker-compose.local.yml up model
```

Then run:

```powershell
python scripts/check_server_health.py
python scripts/run_trace_benchmark.py
```

For a quick functional check that does not wait for trace timestamps:

```powershell
python scripts/run_trace_benchmark.py --no-respect-timestamps --run-id smoke
```

This local score is a proxy. It is not expected to match the leaderboard unless the hardware is comparable to the organizer `1g.18gb` H200 MIG environment.

For GPQA-derived local quality checks, see `docs/baseline/gpqa-benchmark.md`.
