# GPQA Diamond Local Benchmark

This benchmark builds a 120-request OpenAI-compatible trace from GPQA Diamond and replays it through the same docker-compose server used by the baseline.

## Purpose

- Use GPQA Diamond as a local quality proxy before and after serving changes.
- Preserve the arrival schedule from `data/trace-round1.jsonl` so the GPQA trace still exercises the benchmark runner like the competition trace.
- Keep generated GPQA content out of Git. `data/traces/gpqa-*.jsonl` and `data/traces/gpqa-*.answers.json` are ignored.

This does not replace the organizer's private accuracy set. It is a local regression guard for optimization work.

## Inputs

- Dataset: `Idavidrein/gpqa`
- Variant: `gpqa_diamond`
- Count: `120`
- Default seed: `42`
- Schedule source: `data/trace-round1.jsonl`
- Served model name: `Qwen3.5-2B`

The dataset is gated on Hugging Face. Set `HF_TOKEN` after accepting the dataset terms.

## Build The Trace

```powershell
Copy-Item .env.example .env
# Edit .env and set HF_TOKEN=hf_...
python scripts/build_gpqa_trace.py --count 120 --seed 42
```

Outputs:

- `data/traces/gpqa-diamond-120.jsonl`
- `data/traces/gpqa-diamond-120.answers.json`

## Run Against Docker Compose

Start the server:

```powershell
docker-compose -f docker-compose.local.yml up model
```

In another terminal:

```powershell
python scripts/check_server_health.py
python scripts/run_trace_benchmark.py --trace data/traces/gpqa-diamond-120.jsonl --output-root results/gpqa-baseline --run-id baseline
```

Score answer accuracy from the benchmark output:

```powershell
python scripts/evaluate_gpqa_results.py --requests results/gpqa-baseline/baseline/requests.jsonl
```

This writes:

- `results/gpqa-baseline/baseline/summary.json` for ERS/latency proxy
- `results/gpqa-baseline/baseline/requests.jsonl` for per-request latency and model text
- `results/gpqa-baseline/baseline/gpqa_accuracy.json` for local accuracy

Use the same trace and seed for optimized runs, then compare both `summary.json` and `gpqa_accuracy.json`.
