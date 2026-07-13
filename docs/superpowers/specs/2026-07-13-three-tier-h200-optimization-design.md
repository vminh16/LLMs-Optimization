# Three-Tier H200 Optimization Experiment Design

## Goal

Maximize the organizer score for `Qwen/Qwen3.5-2B` on one H200 MIG
`1g.18gb` instance through three progressively more invasive experiments:

1. stock vLLM `v0.22.1` configuration and supported precision controls;
2. an objective-aware serving policy in a thin derived image;
3. profiled, Qwen3.5-specific kernel and memory-layout work.

Each tier must produce a usable finalist on its own. A later tier inherits only
changes that passed the local correctness gate and improved the organizer
score. The design does not require all listed optimization techniques to be
implemented.

## Locked Context and Assumptions

- The organizer entrypoint remains exactly
  `python3 -m vllm.entrypoints.openai.api_server`.
- The organizer model and pinned weights remain unchanged.
- Local development uses an NVIDIA L4. L4 results screen correctness,
  compatibility, and direction; they do not predict the H200 MIG ranking.
- The organizer trace contains 120 requests. The primary local trace is
  `data/trace-round1-diverse-content.jsonl`, SHA-256
  `fcf8992f7a0618f3f7252f706e7999d97bc55380c419a2a03823985f45a8d67f`.
- The current organizer control is `--language-model-only`, score `15.96`,
  TTFT P50 `679 ms`, TTFT P95 `10102 ms`, median TBT `59 ms`, and 84 of 120
  requests with non-zero request score.
- The throughput preset is not a finalist: it increased the effective prefill
  budget from 2048 to 4096 tokens and scored `9.89` despite reducing median TBT
  from 59 to 56 ms.
- Organizer per-request rows are not currently available. Tier A can proceed
  without them. Tier B must use a feedback controller rather than a fitted
  latency model until those rows are available.

## Objective Implications

Within the non-clamped interval, including the request weight of `0.5`, the
absolute marginal utilities are:

```text
|dU_ttft/dTTFT| = (1500 - TTFT) / 1400^2
|dU_tpot/dTPOT| = (45 - TPOT) / 25^2
```

At TTFT `679 ms`, one millisecond is worth approximately `0.000419` request
utility. At TPOT `35 ms`, one millisecond is worth `0.016`, about 38 times as
much. At TPOT `25 ms`, the ratio is about 76. Therefore the scheduler must
protect decode latency before spending extra GPU time on larger prefills.

The squared utility is also convex inside each scoring interval. Saving an
already viable request can be worth more than spending the same GPU time on a
request that will remain beyond a ceiling. Tier B may exploit this property,
while an aging rule still guarantees that every request eventually finishes.

## Existing Mechanisms and Disposition

| Technique | Decision | Reason |
|---|---|---|
| PagedAttention | Keep; no independent experiment | It is already vLLM's cache allocator. |
| Continuous/dynamic batching | Keep; tune its limits | It is already the V1 scheduler behavior. |
| Exact prefix caching | Keep enabled | Local runs reach about 78.5% prefix-hit rate; disabling it caused 52 failures. |
| Semantic response caching | Exclude | Similarity-based response reuse changes task semantics and the local prompts are not exact duplicate requests. |
| FP8 KV cache | Tier A, first priority | It halves full-attention KV bytes and approximately doubles effective hybrid-cache capacity. |
| Online FP8 weights | Tier A, second priority | Hopper has native FP8; per-tensor FP8 is compatible with the Qwen GDN `ba` projection. |
| BF16 GDN SSM state | Tier A, conditional | It halves recurrent-state traffic but overrides the model's explicit FP32 state dtype. |
| CPU/NVMe offload | Exclude unless preemption/OOM is observed | The system has only 3 CPU cores and 8 GB RAM; transfer latency is harmful when FP8 cache already fits. |
| Draft-model speculation | Exclude initially | A second 0.8B model consumes memory and compute under high concurrency. |
| Native MTP-1 | Tier B, conditional branch | The checkpoint has one MTP layer, but vLLM documents lower throughput under high concurrency. |
| FlashAttention/FlashInfer | Keep automatic selection | H200 uses the SM90 FlashInfer GDN-prefill path; forcing a backend can disable a compatible fast path. |
| CUDA graphs | Keep enabled | vLLM already uses full and piecewise graphs; only capture bounds need possible trimming. |

## Experiment A: Stock vLLM Precision and Physical Scheduling

### Purpose

Produce the strongest configuration that can run on the organizer's unchanged
`vllm/vllm-openai:v0.22.1` image. This experiment is a sequential ladder, not
a combinatorial sweep.

### Control A0

Use the current organizer winner without the throughput preset:

```text
--max-model-len=262144
--gpu-memory-utilization=0.95
--tensor-parallel-size=1
--enable-prefix-caching
--language-model-only
```

PagedAttention, continuous batching, chunked prefill, automatic attention
backend selection, and CUDA graphs remain at their vLLM defaults.

### Candidate A1: FP8 KV Cache

Add only:

```text
--kv-cache-dtype=fp8
```

Do not pass `--calculate-kv-scales`. In vLLM `v0.22.1`, the hybrid Qwen3.5
dummy calibration path can observe uninitialized GDN state and corrupt the
derived scales. The safe supported behavior for this model is the default
scale of `1.0`, followed by the GPQA gate.

A1 is mandatory because the previous organizer FP8 result also changed
`max-num-seqs` and therefore does not isolate FP8 KV cache. Existing local
GPQA results of 44, 43, and 43 correct answers are positive evidence but were
generated with the old calibration flag, so A1 must be requalified.

### Candidate A2: Online FP8 Weights

Starting from A1, add only:

```text
--quantization=fp8_per_tensor
```

Do not use `fp8_per_block`: the exact Qwen3.5 GDN implementation states that
`ba_proj` does not support blockwise FP8 quantization. Online per-tensor FP8
quantizes linear weights at load time and dynamically scales activations. It
does not require a new checkpoint or calibration dataset.

### Candidate A3: BF16 GDN Recurrent State

Starting from A2, add:

```text
--mamba-ssm-cache-dtype=bfloat16
```

This candidate is attempted only when A2 loses no more than 2.5 percentage
points of paired local GPQA accuracy relative to the BF16 control. A3 itself
must remain within five points of the BF16 control and within 2.5 points of
A2. Qwen3.5 declares `mamba_ssm_dtype=float32`, so A3 has a higher
numerical-risk class than A1 and A2.

The batch-token budget must preserve comparable physical prefill work. Let
`P` be the attention block size printed by vLLM after hybrid page unification,
and let `N=120` be the maximum number of decode tokens in one scheduler step.
Use:

```text
B(k) = k * P + N
```

With expected page sizes, A1/A2 use `P=1072` and one physical prefill block,
while A3 uses `P=560` and two blocks. Thus A3 uses approximately
`--max-num-batched-tokens=1240`, subject to the startup log confirming the
actual page size. This compares about 1120 A3 prefill tokens with 1072 A2
prefill tokens instead of accidentally allowing three A3 blocks under the
2048-token default.

### Scheduler and CUDA-Graph Finalization

The first organizer submissions keep the proven 2048-token scheduler budget.
After precision is selected:

- set `--max-num-seqs=120`; the trace contains only 120 requests, so this
  cannot reject a trace request and may reduce unnecessary graph-capture
  memory;
- retain the default full-and-piecewise CUDA graph mode;
- do not retest 4096 tokens;
- test a two-page prefill budget `B(2)` only if the winning precision config
  still has median TBT below the score ceiling or organizer request rows show
  adequate TPOT slack.

For FP8 KV with FP32 SSM state, expected `B(2)` is `2264`. This is the only
larger stock scheduler candidate: it corresponds to a physical page boundary,
unlike an arbitrary broad sweep.

Access-log and stats-log flags are operational hygiene, not score-bearing
variables. Omit them from the scored finalist unless they win an independent
measurement.

### A Gates and Run Order

For each candidate:

1. Preflight the exact command and preserve the organizer entrypoint.
2. Start once on L4 and record selected attention/GDN backends, page size,
   available cache bytes, cache-token count, CUDA-graph memory, and any
   preemption.
3. Run the 120-request local trace once. Require 120 successes, exact trace
   hash, no missing usage, and no output/parser corruption.
4. Run paired GPQA against the locked BF16 responses. Require local accuracy
   at least `0.30` and a paired drop no larger than `0.05`. Repeated identical
   GPQA runs are stability checks, not independent statistical samples.
5. Submit A1 independently to the organizer.
6. Submit A2 only if it passes the local gate; submit A3 only if it also passes
   its stricter numerical gate.

The organizer decides promotion. A candidate must have penalty `1`, zero
failures, no warmup requests, and a score higher than its immediate parent.
Aggregate TTFT/TBT explain direction but do not override the score.

The expected stock finalist before measurements is:

```text
--max-model-len=262144
--gpu-memory-utilization=0.95
--tensor-parallel-size=1
--enable-prefix-caching
--language-model-only
--kv-cache-dtype=fp8
--quantization=fp8_per_tensor
--max-num-seqs=120
--max-num-batched-tokens=2048
```

`--mamba-ssm-cache-dtype=bfloat16` and its page-matched token budget are not
part of the expected finalist until GPQA and organizer results validate them.

## Experiment B: Score-Aware Scheduler and Conditional MTP

### Image Boundary

Build a thin image from the exact `vllm/vllm-openai:v0.22.1` base. Copy only a
small Python package that provides a custom scheduler class, selected through
`--scheduler-cls`. Do not fork unrelated vLLM code and do not change the API
server entrypoint.

### B1: Utility- and Cache-Aware Scheduling

In each scheduler iteration:

1. Always reserve one token for every admitted decode request.
2. Maintain an exponential moving average of decode-step latency.
3. Choose zero, one, or two physical prefill pages so predicted step latency
   remains near a `35 ms` TPOT target and below the `45 ms` score ceiling.
4. Rank waiting prefills by predicted request-utility gain per GPU millisecond,
   using cached-token count, remaining prompt tokens, queue age, and current
   page alignment.
5. Prefer a prefix seed when completing it unlocks exact cache blocks for
   multiple queued requests.
6. Stop admitting new decode sequences when their predicted marginal TPOT
   loss exceeds the TTFT utility gained by the next prefill.
7. Age all requests so every one eventually completes after the score-bearing
   window.

Until organizer request rows are available, the controller is feedback-based
and does not fit L4 latency coefficients. If rows become available, replay
them through the local scoring function and replace the heuristic priority
with measured `delta utility / estimated GPU ms`.

### B2: Native MTP-1 Branch

Test MTP only after B1, and only when the Tier A winner still has material
per-request TPOT above `45 ms`, or its reported median TBT remains at or above
`45 ms` when per-request rows are unavailable. Use the checkpoint's single
native MTP layer:

```text
--speculative-config={"method":"mtp","num_speculative_tokens":1}
```

Measure accepted/drafted tokens, TPOT, TTFT, cache usage, and prefix-hit rate.
Reject MTP if acceptance is below 70%, if cache pressure reduces the active
batch, if prefix reuse regresses materially, or if the final score does not
beat B1. Do not test the 0.8B draft model unless MTP-1 proves that speculation
helps this trace but its acceptance quality is the limiting factor.

### B Gates

- The thin image running the unmodified scheduler must reproduce Tier A before
  policy code is enabled.
- No scheduler change may introduce an error, timeout, preemption storm, or
  missing token usage.
- GPQA must remain unchanged for B1. MTP should be output-distribution exact in
  theory, but still runs GPQA because hybrid prefix/speculation bugs are
  possible.
- B1 and B2 are separate organizer submissions. They are not bundled first.

## Experiment C: Profile-Gated Kernel and Layout Work

### Profiling Gate

Do not start with a custom kernel. First collect per-operation GPU time from
the Tier B winner. L4 can validate instrumentation and generic kernels, but an
H200 diagnostic run is required because vLLM selects Triton/FLA GDN prefill on
L4 and FlashInfer GDN prefill on SM90.

For a hotspot fraction `p` and kernel speedup `r`, the maximum end-to-end
speedup is:

```text
speedup = 1 / ((1 - p) + p / r)
```

Proceed only when one target accounts for at least 15-20% of score-relevant
GPU time and the Amdahl bound justifies the build and correctness risk.

### Ordered Kernel Targets

1. Benchmark the existing SM90 FlashInfer GDN-prefill kernel against
   FlashQLA at the model's exact head dimensions, page-aligned chunk sizes,
   and mixed prefill/decode shapes. FlashQLA's published 2-3x result is versus
   FLA Triton, not the H200 path already selected by vLLM.
2. If decode is dominant, optimize the packed recurrent GDN update: BF16 state
   storage with FP32 accumulation, fused conv/state update/gating, and fewer
   intermediate writes. Validate recurrent error over long prompts before
   GPQA.
3. If launch and layout overhead is dominant, fuse only the measured adjacent
   operations and specialize CUDA-graph capture for the observed active batch
   sizes.
4. Consider decoupling attention-page and GDN-state granularity only if page
   alignment is measurably reducing prefix reuse or causing scheduler stalls.

Full-attention kernels remain FlashAttention/FlashInfer unless profiling shows
they dominate after FP8 KV. A framework rewrite, INT4 path, semantic cache, or
NVMe cache is outside Tier C because none has a stronger expected score gain
than the ordered targets on Hopper MIG.

### C Gates

- Build for SM90 while retaining a functional L4 fallback.
- Compare kernel outputs with the stock implementation over prefill/decode,
  page boundaries, mixed batches, and long recurrent sequences.
- Require the same 120-request success contract and paired GPQA gate as Tier A.
- Promote only an end-to-end organizer score improvement; microbenchmark
  speedup alone is insufficient.

## Evidence Sources

- vLLM `v0.22.1` Qwen GDN implementation and backend selection:
  <https://github.com/vllm-project/vllm/blob/v0.22.1/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py>
- vLLM online quantization:
  <https://docs.vllm.ai/en/v0.22.0/features/quantization/online/>
- vLLM automatic prefix caching:
  <https://docs.vllm.ai/en/v0.22.1/features/automatic_prefix_caching/>
- vLLM Qwen3.5 recipe and MTP trade-off:
  <https://github.com/vllm-project/recipes/blob/main/Qwen/Qwen3.5.md>
- Qwen3.5-2B model configuration:
  <https://huggingface.co/Qwen/Qwen3.5-2B/blob/main/config.json>
- PagedAttention:
  <https://arxiv.org/abs/2309.06180>
- Sarathi-Serve chunked-prefill scheduling:
  <https://arxiv.org/abs/2403.02310>
- FlashQLA:
  <https://qwen.ai/blog?id=flashqla>
- Hybrid FP8 KV scale failure:
  <https://github.com/vllm-project/vllm/issues/37554>

## Non-Goals

- Do not repeat every possible vLLM flag or combine unqualified candidates.
- Do not treat L4 absolute latency or ERS as an H200 prediction.
- Do not force a backend merely because it is newer.
- Do not add offload, semantic caching, a second draft model, or custom CUDA
  without its stated activation condition.
- Do not edit the current submission compose while designing the experiments.
