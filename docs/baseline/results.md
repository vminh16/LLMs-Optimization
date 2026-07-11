# Locked Baseline Results

## H0.1 Transition

New performance experiments use `data/trace-round1-diverse-content.jsonl` and
the H0.1 measurement contract. The locked results below used the legacy trace
and pre-H0.1 client behavior, so they remain historical evidence and must not
be compared numerically with new candidates. Lock a new three-run cold
baseline under `results/trace-baseline-h01` before Experiment 1.

## Baseline Identity

- Organizer compose: accepted, leaderboard score approximately `14`.
- Local GPU: NVIDIA L4.
- Serving image: `vllm/vllm-openai:v0.22.1`.
- Model: `Qwen/Qwen3.5-2B`, default Hugging Face revision.
- Legacy performance trace: `data/trace-round1.jsonl`, 120 requests.
- Quality trace: public GPQA Diamond sample, 120 questions, seed 42.

The organizer score is the official reference. L4 results are relative
screening measurements and cannot be converted to an H200 MIG score.

## L4 Performance Baseline

All three runs completed 120 requests with zero errors.

| Run | Median TTFT | P95 TTFT | Median TPOT | P95 TPOT | Makespan | Output chunks/s |
|---|---:|---:|---:|---:|---:|---:|
| baseline-01 | 21329.47 ms | 51667.18 ms | 269.86 ms | 337.75 ms | 97859.31 ms | 244.77 |
| baseline-02 | 21681.99 ms | 51973.47 ms | 270.32 ms | 339.41 ms | 98173.52 ms | 244.47 |
| baseline-03 | 19211.67 ms | 49682.77 ms | 271.65 ms | 340.18 ms | 95554.40 ms | 251.17 |
| **Three-run median** | **21329.47 ms** | **51667.18 ms** | **270.32 ms** | **339.41 ms** | **97859.31 ms** | **244.77** |

These results were produced before H0 measurement standardization, when the
local runner counted non-empty streaming chunks. New benchmark runs should use
`usage.completion_tokens` and TPOT from `(total_latency_ms - TTFT_ms) /
(completion_tokens - 1)`.

Official ERS is saturated near zero on L4:

- baseline-01: `0`
- baseline-02: `0`
- baseline-03: `0.0001378285`

Run 03 received a small non-zero score because one request had TTFT below the
1500 ms ceiling. Raw TTFT, TPOT, makespan, and output rate remain the local
performance decision metrics.

## Local Accuracy Baseline

All three GPQA runs produced the same result:

- Correct: `43 / 120`
- Accuracy: `0.3583333333`
- Missing answers: `0`
- Local baseline delta: `0`
- Local accuracy multiplier: `1`

The official competition reference remains `0.40`. The local evaluator uses
`43 / 120` because the public sample and model revision differ from the
organizer evaluation.

With the same penalty shape, the discrete local gate is:

| Correct answers | Local multiplier |
|---:|---:|
| 31-120 | 1.0 |
| 24-30 | Linear penalty |
| 0-23 | 0.0 |

Configuration-only candidates should remain at 43/120. The wider local gate is
for numerical changes such as quantization, not permission to ignore a quality
regression.

## Mathematical Priority

Inside the active scoring interval, a score term is:

```text
s(x) = ((C - x) / (C - F))^2
ds/dx = -2(C - x) / (C - F)^2
```

At the midpoint of each interval:

- weighted TTFT slope: about `-0.000357` score per millisecond;
- weighted TPOT slope: `-0.02` score per millisecond.

Near the midpoints, one millisecond of TPOT is therefore worth roughly 56
milliseconds of TTFT. TPOT is the first scoring target, while TTFT remains a
constraint because excessive queueing can erase its half of the score.

The trace supports this diagnosis:

- all 120 requests share one exact 38,956-character system prompt;
- there are 20 exact user prompts, each repeated six times;
- requests arrive over 25.475 seconds, peaking at six requests per second;
- later duplicate occurrences wait longer, while TPOT falls as active decode
  concurrency drains.

Automatic prefix caching is already enabled and matches the workload. The
remaining large opportunity is scheduler balance between prefill admission,
active decode concurrency, and queue delay.

## Optimization Order

### 1. Free Wins

1. Keep prefix caching enabled and test `--prefix-caching-hash-algo=xxhash`.
   The workload has unusually large exact prefixes and the grading host has
   only three CPU cores.
2. Test FP8 KV cache as a low-effort memory/bandwidth candidate, but require
   GPQA because KV-cache quantization changes numerical behavior.
3. Record startup-resolved scheduler values before making them explicit.
4. Confirm whether async scheduling and chunked prefill are already active.
5. Measure exact token lengths before reducing `--max-model-len`; never choose
   a limit from character counts.

### 2. Simple Configuration Optimization

1. Sweep `--max-num-seqs` first to limit simultaneous decodes and reduce TPOT.
2. At the best sequence limit, sweep `--max-num-batched-tokens`. vLLM documents
   smaller budgets as favoring inter-token latency and larger budgets as
   favoring TTFT and throughput.
3. Keep one variable per experiment, run each candidate three times, and
   compare medians with the locked baseline.
4. Re-run GPQA for changes involving prefix caching or chunked prefill because
   this model uses hybrid Gated DeltaNet attention.

Suggested first coarse sweep:

```text
max_num_seqs: 8, 16, 32, 64
max_num_batched_tokens: 2048, 4096, 8192
```

Only perform the token-budget sweep around the best sequence setting. Do not
run the full Cartesian product initially.

### 3. Precision And Memory Optimization

1. Test an offline FP8 W8A8 checkpoint before INT4. Both L4 and H200 support
   FP8, while FP8 generally carries less quality risk. vLLM notes that online
   dynamic FP8 has limited latency gains, so it is a smoke candidate rather
   than the final quantization path.
2. If FP8 KV cache passes the free-win gate, combine it with the best scheduler
   config later; if it fails, revisit it only with calibrated scales.
3. Require GPQA evaluation for every precision candidate.

### 4. Complex Optimization

1. Speculative decoding with the same-family 0.8B draft model.
2. MTP or another trained speculator only after measuring draft acceptance.
3. Custom Triton/CUDA kernels only after profiling identifies a specific
   dominant operation.

Speculative decoding is not an early default: vLLM positions it primarily for
medium-to-low QPS, memory-bound decode workloads, while this trace creates a
high-load queue and the draft model consumes scarce 18 GB submission memory.

## Experiment Acceptance

A local candidate advances only when:

1. all 120 requests succeed;
2. three-run median TPOT improves beyond run-to-run noise;
3. TTFT P95 and makespan do not regress enough to cancel the decode gain;
4. configuration-only GPQA remains 43/120;
5. numerical candidates retain a safe local multiplier;
6. the final candidate is confirmed by an organizer submission.

Official vLLM references:

- [v0.22.1 engine arguments](https://docs.vllm.ai/en/v0.22.1/configuration/engine_args/)
- [v0.22.1 optimization and chunked-prefill tuning](https://docs.vllm.ai/en/v0.22.1/configuration/optimization/)
- [v0.22.1 FP8 W8A8](https://docs.vllm.ai/en/v0.22.1/features/quantization/fp8/)
- [v0.22.1 quantized KV cache](https://docs.vllm.ai/en/v0.22.1/features/quantization/quantized_kvcache/)
- [speculative decoding](https://docs.vllm.ai/en/stable/features/spec_decode/)
