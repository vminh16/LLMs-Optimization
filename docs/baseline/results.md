# Baseline And Harness Status

## Current Position

- Organizer baseline remains the best confirmed submission: `15.19`.
- Local development GPU is NVIDIA L4; local numbers are relative rankings, not
  estimates of the organizer H200 MIG score.
- The active performance reference is `results/trace-baseline-h01` using
  `data/trace-round1-diverse-content.jsonl`.
- The independent local quality reference is GPQA Diamond: `43 / 120`.
- Experiment 1 is the active phase; no candidate has been promoted yet.

## Completed Harness Phases

| Phase | Result |
|---|---|
| Baseline | Organizer compose accepted; local legacy performance and GPQA baselines recorded. |
| H0 | Audited the original harness and identified client queuing, chunk-count token estimates, missing trace identity, and noisy readiness. |
| H1 | Implemented the H0.1 contract: 120 HTTP connections, exact streaming usage, dispatch telemetry, trace SHA-256, and stable readiness. |
| H2 | Locked three cold H0.1 runs and verified repeatability; this is the comparison baseline for new experiments. |

## Locked H0.1 Median

All three runs completed `120 / 120` requests with zero failures and the same
trace hash.

| Metric | Median |
|---|---:|
| TTFT P50 | 24,735.54 ms |
| TTFT P95 | 59,283.24 ms |
| TBT median | 281.41 ms |
| Makespan | 106,817.57 ms |
| Dispatch lag P95 | 8.59 ms |

Official ERS is saturated at zero on L4, so local decisions use relative TBT,
TTFT P95, and makespan. Do not compare H0.1 candidates with legacy artifacts
under `results/trace-baseline` or `results/trace-baseline-h0`.

## Official Free-Win Evidence

| Submission | Score |
|---|---:|
| BF16 baseline | 15.19 |
| Without `max-num-seqs=64` | 9.31 |
| FP8 KV + `max-num-seqs=64` | 9.37 |
| FP8 KV + `max-num-seqs=64` repeat | 9.23 |

These submissions do not support promoting FP8 KV or the tested scheduler
bundle. FP8 KV remains parked until it wins independently on organizer-like
hardware and passes GPQA.
