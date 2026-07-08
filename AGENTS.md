# AGENTS.md — Immutable First-Read Rules

This file is the repository's first-read operating rule for coding agents. Treat it as immutable. Do not edit, weaken, bypass, or reinterpret these rules unless the user explicitly asks to change this file.

This file has two parts: **Part A (Operating Rules)** governs *how* you work and does not change. **Part B (Project Specification)** describes *what* you are building — it is fixed for the duration of Phase 1 (it is the competition's rules, not your own design, so don't reinterpret it either), except for the **Team Decisions Log** at the very end of Part B, which is the one section you should actively update as choices are made and validated.

If `GEMINI.md`, `CLAUDE.md`, `.cursor/rules/`, or `SPEC.md` exist elsewhere in this repo, read them too and merge with this file. Right now no separate `SPEC.md` exists — Part B below *is* the spec. If one is added later, treat it as authoritative for spec details and this file as authoritative for process/rules; where they conflict, follow the more cautious and more specific instruction.

---

## Part A — Operating Rules

### 1. Think Before Coding
Do not assume. Do not hide confusion. Surface tradeoffs.

Before implementing:
- State assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them instead of silently choosing.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop, name what is confusing, and ask.

### 2. Simplicity First
Write the minimum code that solves the problem. Add nothing speculative.
- No features beyond what was asked.
- No abstractions for single-use code.
- No flexibility or configurability that was not requested.
- No error handling for impossible scenarios.
- If 200 lines could safely be 50, rewrite it.
- Ask: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes
Touch only what you must. Clean up only your own mess.

When editing existing code:
- Do not improve adjacent code, comments, or formatting.
- Do not refactor things that are not broken.
- Match existing style, even if you would do it differently.
- If you notice unrelated dead code, mention it. Do not delete it.

When your changes create orphans:
- Remove imports, variables, functions, and files that your change made unused.
- Do not remove pre-existing dead code unless asked.

The test: every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution
Define success criteria. Loop until verified.

Transform tasks into verifiable goals:
- "Add validation" means write tests for invalid inputs, then make them pass.
- "Fix the bug" means write a test that reproduces it, then make it pass.
- "Refactor X" means ensure tests pass before and after.

For multi-step tasks, state a brief plan:
1. Step -> verify: check.
2. Step -> verify: check.
3. Step -> verify: check.

Strong success criteria let you loop independently. Weak criteria like "make it work" require clarification before implementation.

These guidelines are working when diffs are smaller, unnecessary changes are rarer, rewrites from overcomplication decrease, and clarifying questions happen before implementation mistakes.

---

## Part B — Project Specification

### B0. What this repo is
Competition submission for **Viettel AI Race 2026**, track: **LLM Inference Optimization (Phase 1 / online round)**. The task is a systems-optimization problem, not a modeling problem: deploy and tune an inference server for a *given, fixed* model so it serves a fixed request trace with the best possible mix of throughput, latency, and output quality, under tight, non-negotiable hardware limits.

**Objective in one sentence:** maximize `Score = 100 × ERS × f(Δ)` (defined in B4) for `trace-round1.jsonl` (120 requests), by editing the serving configuration and/or the Docker image — not by changing what problem is being solved.

### B1. Hardware & runtime environment — read this before assuming anything about capacity
- **GPU: 1× MIG H200 instance, profile `1g.18gb`.** This is the *smallest* MIG slice NVIDIA defines on a 141GB H200 — roughly **14% of the chip's compute (SMs)** and **18GB of the 141GB total memory**. Memory bandwidth is partitioned proportionally too (MIG splits both compute and memory controllers), so do not use full-H200 numbers (2× H100 memory, 4.8TB/s bandwidth) as a reference point for anything — the real available bandwidth on this slice is closer to a mid-range datacenter GPU.
- **3 CPU cores, 8GB system RAM.** Both are small. CPU-side overhead (tokenization, request scheduling, Python dispatch) is a more plausible bottleneck here than in a normal multi-core server — do not dismiss it.
- **OS/driver: Ubuntu 22.04 LTS, CUDA 12.x.**
- **Do not assume you have this exact hardware available for local development.** Ask the user what dev/test hardware is actually available (full GPU rental, a different MIG slice, a consumer GPU, CPU-only) before writing tuning code that hardcodes assumptions calibrated to `1g.18gb`. Numbers tuned on a different GPU shape (different compute:VRAM:bandwidth ratio) will not transfer reliably — flag this explicitly if dev hardware differs from the grading hardware.

### B2. The model — do not treat this as a plain Transformer
**Model: `Qwen/Qwen3.5-2B`, BF16, weights pinned by a fixed HF Hub hash the organizers provide.**

This is labeled "Dense Transformer" in the brief, which is true only in the sense that it is *not* MoE (all parameters active). It is **not** a standard softmax-attention Transformer:

- It uses a **hybrid Gated DeltaNet architecture**: roughly 75% of layers are linear-attention (DeltaNet, fixed-size recurrent state, not a growing KV-cache) and ~25% are full softmax attention with GQA+RoPE (every 4th layer, roughly). This is *why* a 2B model can support the 262144-token context length configured in the baseline without an unreasonable memory footprint — most layers don't need a cache that grows with sequence length.
- **Practical consequence you must verify empirically, not assume:** vLLM's support for hybrid/linear-attention model families (this one included) has been under active development. Features like automatic prefix caching and chunked prefill for hybrid attention models have specifically been called out by the vLLM team as areas of recent/ongoing work for this architecture family. **Before trusting any change to `--enable-prefix-caching` or any chunked-prefill setting, verify output *correctness* (not just speed) against the accuracy eval (B7). A regression here can silently corrupt output while still returning HTTP 200 — it will not show up as an error, it will show up as a wrecked accuracy score.**
- Do not assume MoE-specific techniques (expert parallelism, expert load balancing) apply — this model does not use MoE.
- A same-family, same-tokenizer smaller model (`Qwen3.5-0.8B`, from the same release) is a natural candidate as a speculative-decoding draft model if that avenue is explored — no need to source or train an unrelated draft model.

### B3. Given baseline — starting point, not a suggestion
This exact Docker image and command are the organizer-provided baseline. Reproduce it first, get it submitted and scored once, *then* start changing things — don't optimize blind before you have one real leaderboard data point.

```yaml
services:
  model:
    image: vllm/vllm-openai:v0.22.1
    entrypoint:
      - python3 #Don't change this to vllm-server
      - -m  #Don't change this to vllm-server
      - vllm.entrypoints.openai.api_server #Don't change this to vllm-server
    command:
      - --model=/model #Don't change this to vllm-server
      - --served-model-name=Qwen3.5-2B #Don't change this to vllm-server
      - --host=0.0.0.0 #Don't change this to vllm-server
      - --port=8000 #Don't change this to vllm-server
      - --max-model-len=262144
      - --gpu-memory-utilization=0.95
      - --tensor-parallel-size=1
      - --enable-prefix-caching
    ports:
      - "8000:8000"
    shm_size: "2g"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

**The repeated `#Don't change this to vllm-server` comments are a hard constraint from the organizers, not a stray annotation — preserve the `python3 -m vllm.entrypoints.openai.api_server` invocation exactly.** The grading harness likely depends on this exact entrypoint shape (healthcheck, log parsing, or process signaling). If you believe changing it is necessary, stop and ask the user first — do not silently "improve" it to the `vllm serve` shorthand.

Already configured in baseline: prefix caching on, 95% GPU memory utilization, TP=1, 256K max context. **Not yet configured: any quantization.** Baseline runs full BF16.

### B4. Scoring — the actual objective function; read before writing any optimization code

```
Score = 100 × ERS × f(Δ)
```

This is a **product**, not a sum. `f(Δ) = 0` zeroes the entire score regardless of how good ERS is. **Correctness/accuracy safety gates everything — validate it before and after every serving change, not just at the end.**

**ERS (Effective Request Score)**, averaged over all 120 requests in `trace-round1.jsonl`:
```
S_request = 0                                  if error / timeout / 0 tokens returned
S_request = w · s_ttft + (1 − w) · s_tpot      if successful

s_ttft = clamp((C_ttft − TTFT) / (C_ttft − F_ttft), 0, 1) ^ γ
s_tpot = clamp((C_tpot − TPOT_mean) / (C_tpot − F_tpot), 0, 1) ^ γ
```

| Param | Meaning | Value |
|---|---|---|
| F_ttft | TTFT floor (perfect score at/below this) | 100 ms |
| C_ttft | TTFT ceiling (zero score at/above this) | 1500 ms |
| F_tpot | TPOT floor | 20 ms |
| C_tpot | TPOT ceiling | 45 ms |
| γ | exponent | 2 |
| w | weight on TTFT (vs TPOT) | 0.5 |

**Read the shape of this function before tuning, it is not intuitive:**
- TPOT's floor-to-ceiling range (20–45ms) is much narrower than TTFT's (100–1500ms), and squaring (`γ=2`) punishes the middle of the range hard. Concretely: TPOT=25ms scores 0.64 on that term, TPOT=35ms scores only 0.16 — a 10ms difference costs ¾ of the credit. **TPOT is the fragile metric, not TTFT.**
- Higher concurrency/batch size generally helps admit requests sooner (can help TTFT) but tends to raise mean per-token latency for in-flight requests (hurts TPOT) because decode steps share the same limited compute/bandwidth of the `1g.18gb` slice across more sequences. **Pushing concurrency past the point where TPOT crosses ~35-40ms can lower the total score even while "throughput" (raw requests/sec) goes up.** The right concurrency is an empirical sweet spot on *this* hardware and *this* trace, not a value to maximize.

**Accuracy Gate (GPQA Diamond, 100 fixed questions, independent of the trace):**
```
Δ = baseline_accuracy − your_accuracy        (baseline_accuracy = 0.40)

f(Δ) = 1.0                              if Δ ≤ 0.10   (accuracy ≥ 30%: full credit, no penalty)
f(Δ) = 1.0 − (Δ − 0.10) / 0.06          if 0.10 < Δ < 0.16   (linear ramp down)
f(Δ) = 0.0                              if Δ ≥ 0.16   (accuracy ≤ 24%: score is zero, period)
```
There is real slack here (up to 10 points of degradation is free), but the failure mode past 16 points is total, not graceful. **Never accept a quantization/optimization change without measuring its actual Δ on a real GPQA Diamond eval run** — do not infer safety from what a technique "usually" costs on other models/benchmarks.

### B5. Optimization scope — anything below is allowed; none of it is required
Framework choice is open (vLLM, SGLang, TensorRT-LLM, custom) but see B3 — switching away from the working, provided vLLM baseline has a real cost for a team without prior framework experience; treat that as a considered decision, not a default.

- **Quantization** — weight quantization (FP8 `F8_E4M3`, INT8, INT4, mixed-precision, AWQ, GPTQ), activation/dynamic quantization.
- **KV Cache & Memory** — PagedAttention (already vLLM's default memory manager), KV-cache quantization (FP8, INT8), prefix caching / semantic caching, CPU/NVMe offloading.
- **Serving & Scheduling** — dynamic/continuous batching, speculative decoding (draft model or self-speculative), memory-aware scheduling.
- **System & Runtime** — custom CUDA/Triton kernels, fused attention kernels (FlashAttention, FlashInfer), memory layout / CUDA Graph optimization.

Nothing here is scored for being "used" — only the final `Score` matters. Do not add a technique to check a box; add it if it measurably improves `Score` on real evaluation runs.

### B6. Submission workflow
1. Build the Docker image with your changes.
2. Push it **public** to Docker Hub.
3. Submit a `docker-compose.yml` pointing at that image via the organizer Portal.
4. Automated grading: they pull the image, run it on a real `1g.18gb` instance, healthcheck it, then benchmark against the (private, on their side) trace.
5. Leaderboard updates automatically, results/logs back in ~15 minutes.

This is a fast feedback loop — use it. Submitting the untouched baseline first, to get one real scored data point and confirm the pipeline works end to end, is higher priority than any optimization work.

### B7. Accuracy evaluation harness
Use the public **GPQA Diamond** dataset (`Idavidrein/gpqa` on Hugging Face, gated — requires accepting usage terms) as a local proxy for the organizers' 100-question eval. It won't be the identical question set, but it's the same distribution and is the right tool to catch a quantization method that's about to blow through the Δ≥0.16 cliff. Building/maintaining this harness and running it against every candidate serving config is not optional cleanup work — treat it as equal priority to the serving code itself, per B4.

### B8. Do not assume — ask instead
- The actual arrival pattern, prompt-length distribution, and output-length distribution of `trace-round1.jsonl` — read the file before tuning concurrency/batching for it.
- Whether the GPQA accuracy numbers you get locally are being computed the same way the organizers compute theirs (prompt format, few-shot vs zero-shot, answer-extraction method) — mismatched eval methodology gives a false sense of safety margin.
- What hardware is actually available for local iteration, and whether it's a reasonable stand-in for `1g.18gb` (see B1).
- Docker Hub namespace/credentials to push to — do not invent one.
- Whether the team has already picked a quantization method or framework before you start — check the Team Decisions Log below first.

### B9. Team Decisions Log — the one section you should update
*(Keep entries short: date, decision, measured result, who validated it. This is the only living part of this file — everything else in Part B is the fixed competition spec.)*

- _(empty — first entry goes here once the baseline is submitted and scored)_