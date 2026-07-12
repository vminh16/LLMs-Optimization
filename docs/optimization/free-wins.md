# Optimization Experiments

## Current Status

- Baseline and harness phases H0-H2 are complete.
- The locked local reference is `results/trace-baseline-h01`.
- Organizer BF16 baseline scored `15.19`; `language-only` is the current
  confirmed MVP at `15.96`.
- FP8 KV and the tested `max-num-seqs=64` bundle lost officially and are not MVP candidates.
- Experiment 1 is active and keeps model weights/cache in BF16.

## Experiment 1

Independent candidates:

| Symbol | Candidate |
|---|---|
| B | BF16 baseline reference |
| L | `--language-model-only` |
| P-I | `--performance-mode=interactivity` |
| P-T | `--performance-mode=throughput` |
| C-off | prefix caching off; provisional until GPQA |
| LA | `--disable-uvicorn-access-log` |
| LS | `--disable-log-stats` |
| MM0 | `--mm-processor-cache-gb=0`; renderer control |
| R | MM0 + `--renderer-num-workers=2`; compare only with MM0 |

R contains the MM0 compatibility flag because vLLM rejects concurrent
renderer workers while the multimodal processor cache is enabled. This is a
controlled comparison, not an Experiment 2 optimization bundle. Every other
candidate compares with the locked BF16 baseline.

One screening run is enough to reject a candidate. A finalist requires three
cold runs. A candidate advances when TBT improves at least 1% or TTFT P95 at
least 3%, while makespan does not regress more than 2%.

## Linux VM Procedure

Use Docker Compose v2:

```bash
python --version
docker compose version
nvidia-smi
docker run --rm --gpus all --entrypoint python3 vllm/vllm-openai:v0.22.1 \
  -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Preflight all candidates without loading model weights. This validates Docker,
the resolved Compose command, advertised vLLM flags, local model files, and GPU
visibility:

```bash
python scripts/run_serving_sweep.py --suite experiment1 --preflight-only
```

The existing `language-only` runs remain valid. Screen the remaining independent
candidates one at a time; this makes the failing config obvious and avoids
loading the model for the next candidate after a failure:

```bash
python scripts/run_serving_sweep.py --suite experiment1 --candidate performance-interactivity --repeat 1
python scripts/run_serving_sweep.py --suite experiment1 --candidate performance-throughput --repeat 1
python scripts/run_serving_sweep.py --suite experiment1 --candidate disable-access-log --repeat 1
python scripts/run_serving_sweep.py --suite experiment1 --candidate disable-log-stats --repeat 1
python scripts/run_serving_sweep.py --suite experiment1 --candidate prefix-off --repeat 1
python scripts/run_serving_sweep.py --suite experiment1 --candidate mm-cache-off --repeat 1
python scripts/run_serving_sweep.py --suite experiment1 --candidate renderer-2 --repeat 1 --force
python scripts/summarize_experiment.py --min-runs 1
```

Run MM0 before R. The one-time `--force` for `renderer-2-01` is required because
the earlier failed artifact used the invalid command without MM0. Do not use
`--force` for completed candidates.

Promote only finalists to three runs:

```bash
python scripts/run_serving_sweep.py \
  --suite experiment1 \
  --candidate <winner> \
  --repeat 3 \
  --resume
python scripts/summarize_experiment.py --min-runs 3
```

Preflight proves CLI/environment compatibility only. Model-load OOM, CUDA
graph compilation, startup duration, and real performance must be checked by a
run on the VM. Failed runs retain `experiment.json` and `docker.log` under
`results/experiment1/<candidate>-NN`.

## Experiment 1 Pass Conditions

- **Config gate:** preflight exits `0`, resolved entrypoint remains the organizer
  entrypoint, and no conflicting/default-incompatible flags are present.
- **Startup gate:** the model container stays `running` throughout the grace
  period and `/v1/models` returns `Qwen3.5-2B` twice consecutively. A stopped,
  dead, restarting, or missing container fails immediately.
- **Measurement gate:** exactly 120 requests, zero failures, matching trace hash,
  and measurement version `h0.1`.
- **Performance gate:** advance when median TBT improves at least 1% or TTFT P95
  improves at least 3%, while makespan regresses no more than 2%. Three cold runs
  are required before promotion.
- **Control gate:** R is blocked until MM0 has the same trace identity and enough
  valid runs. C-off remains provisional until GPQA passes.
- **Final gate:** local results screen crashes and large regressions; the
  organizer result decides ranking for CPU/frontend flags because the local L4
  harness did not reproduce the official gain from L.

## Experiments After MVP

Use the winning Experiment 1 config as the new reference and keep one new
variable per gate.

1. **E2.1 - Scheduler (`S`)**: sweep `max-num-seqs`, then sweep
   `max-num-batched-tokens` only around the best sequence limit.
2. **E2.2 - Pairwise**: test `S+L` and `S+R`. Test `S+K` only if FP8 KV (`K`)
   is independently requalified; current official evidence parks it.
3. **E2.3 - Low-risk bundle**: test `S+L+R` from independently winning parts.
4. **E2.4 - Precision bundle**: test `S+L+R+K` only if `K` wins independently
   and passes GPQA.
5. **E3 - Speculation (`M`)**: add MTP-1 to the final non-speculative config;
   accept it only after measuring acceptance, TBT, TTFT, memory, and GPQA.

Prefix-off joins a bundle only if it wins Experiment 1 and passes GPQA.
Quantization, FP8 KV, prefix/chunked-prefill behavior, MTP, and framework
changes always require the independent GPQA harness before submission.
