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
docker-compose -f docker-compose.local.yml -f configs/experiments/freewin-prefix-xxhash.compose.yml up model
```

or:

```powershell
docker-compose -f docker-compose.local.yml -f configs/experiments/freewin-kv-fp8.compose.yml up model
```

Then run the same trace benchmark:

```powershell
python scripts/check_server_health.py
python scripts/run_trace_benchmark.py --trace data\trace-round1.jsonl --output-root results\trace-freewins --run-id prefix-xxhash-01
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
