# Free-Win Optimization Candidates

These candidates are intentionally small compose overrides. They do not replace
`docker-compose.yml` until a benchmark result proves they are better than the
locked baseline.

## Candidates

### Prefix Cache Hashing

File: `configs/experiments/freewin-prefix-xxhash.compose.yml`

This keeps prefix caching enabled and changes only the prefix-cache hash from
the default SHA-256 path to `xxhash`. The trace has very large repeated prefixes
and the target environment has only three CPU cores, so reducing hash overhead
is a reasonable first free win.

Risk: `xxhash` is non-cryptographic and vLLM documents a theoretical collision
risk. This is acceptable only as a benchmark candidate, not as a general
multi-tenant serving default.

### FP8 KV Cache

File: `configs/experiments/freewin-kv-fp8.compose.yml`

This changes only KV-cache storage to FP8 and enables dynamic KV scale
calculation. It is a low-effort memory/bandwidth candidate, but it is not
accuracy-free: cache quantization changes numerical behavior and must pass GPQA.

Risk: Qwen3.5-2B is a hybrid Gated DeltaNet model, so only part of the model
uses a growing full-attention KV cache. The win may be smaller than on a plain
Transformer, and accuracy can regress silently.

## Run

Start one candidate at a time:

```powershell
docker compose -f docker-compose.local.yml -f configs/experiments/freewin-prefix-xxhash.compose.yml up model
```

or:

```powershell
docker compose -f docker-compose.local.yml -f configs/experiments/freewin-kv-fp8.compose.yml up model
```

Then run the same trace benchmark:

```powershell
python scripts/check_server_health.py
python scripts/run_trace_benchmark.py --trace data\trace-round1-diverse-content.jsonl --output-root results\trace-freewins --run-id prefix-xxhash-01
```

For FP8 KV cache, also run GPQA before accepting the candidate:

```powershell
python scripts/run_trace_benchmark.py --trace data/traces/gpqa-diamond-120.jsonl --output-root results/gpqa-freewins --run-id kv-fp8-01
python scripts/evaluate_gpqa_results.py --requests results/gpqa-freewins/kv-fp8-01/requests.jsonl
```

## Acceptance

A candidate advances only if all 120 trace requests succeed, median TPOT or
makespan improves beyond baseline noise, and GPQA remains safe. Prefix hashing
should keep the local GPQA result at `43 / 120`; FP8 KV cache may use the local
accuracy gate but should not be promoted on speed alone.

## Experiment 1: Independent BF16 Flags

Experiment 1 screens five independent changes without FP8 KV or scheduler
tuning: language-model-only, renderer workers 2, both vLLM performance modes,
and prefix caching off. The locked H0.1 result under
`results/trace-baseline-h01` is the comparison baseline and is not rerun by
default.

### Linux VM Gate

Use Docker Compose v2 (`docker compose`), not the legacy
`docker-compose` 1.x binary. Check the VM before running the suite:

```bash
python --version
docker compose version
nvidia-smi
docker run --rm --gpus all --entrypoint python3 vllm/vllm-openai:v0.22.1 \
  -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The GPU probe must print `True` and the expected GPU name. Then validate each
candidate's resolved Compose command and vLLM CLI without loading model
weights:

```bash
python scripts/run_serving_sweep.py --suite experiment1 --preflight-only
```

Preflight proves environment and CLI compatibility only. It cannot detect
model-load OOM, CUDA graph compilation failures, or actual startup time. Use
one real candidate as a canary before the full sweep:

```bash
python scripts/run_serving_sweep.py --suite experiment1 --candidate language-only --repeat 1
```

If the canary completes, continue the remaining candidates while preserving
its valid artifact:

```bash
python scripts/run_serving_sweep.py --suite experiment1 --repeat 1 --resume
```

Preview every Docker command without creating artifacts or starting Docker:

```powershell
python scripts/run_serving_sweep.py --suite experiment1 --dry-run
```

Verify Docker, resolved Compose, image CLI flags, the local model, and GPU
access without loading model weights:

```powershell
python scripts/run_serving_sweep.py --suite experiment1 --preflight-only
```

Run one cold screening measurement per candidate:

```powershell
python scripts/run_serving_sweep.py --suite experiment1 --repeat 1 --output-root results/experiment1
```

Run or resume a single finalist. Use `--force` only to discard the known
artifacts for that exact run ID and repeat it from scratch.

```powershell
python scripts/run_serving_sweep.py --suite experiment1 --candidate renderer-2 --repeat 3 --resume
```

Compare candidates with the H0.1 medians:

```powershell
python scripts/summarize_experiment.py --min-runs 1
```

A candidate advances when median TBT improves by at least 1%, or TTFT p95 by
at least 3%, while makespan does not regress by more than 2%. It is rejected
when TBT regresses by at least 2%, makespan by at least 3%, or TTFT p95 by at
least 7%; smaller changes are inconclusive. A finalist needs three clean runs.
Prefix-off remains provisional until GPQA passes because Qwen3.5 is a hybrid
attention model and the local trace is not the organizer's private trace.

Each run stores the exact command, trace/config hashes, status, summary,
requests, and Docker diagnostics under `results/experiment1/<candidate>-NN`.
The runner waits for stable readiness, refuses silent overwrites, and always
stops Compose after success or failure.
