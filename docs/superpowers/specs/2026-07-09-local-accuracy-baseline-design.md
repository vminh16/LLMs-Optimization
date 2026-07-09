# Local Accuracy Baseline Design

## Goal

Calibrate the local GPQA accuracy penalty to the measured public-dataset
baseline without changing the official competition scoring definition.

## Decisions

- Keep the official `baseline_accuracy=0.40` defaults in benchmark scoring.
- Define the local GPQA reference as `43 / 120` (`0.358333...`).
- Reuse the official penalty shape with the local reference:
  - full multiplier through a 0.10 accuracy drop;
  - linear penalty for a drop between 0.10 and 0.16;
  - zero multiplier at a drop of 0.16 or greater.
- Let the GPQA evaluator accept an optional `--baseline-accuracy`; default it
  to the local reference.
- Add `baseline_accuracy`, `accuracy_delta`, and `accuracy_multiplier` to the
  GPQA evaluation report.
- Keep official ERS thresholds unchanged. L4 optimization comparisons use raw
  TTFT, TPOT, throughput, and errors because official ERS is saturated near
  zero on this hardware.

## Data Flow

1. `score_gpqa_results` computes accuracy from the fixed local answer key.
2. `evaluate_gpqa_results.py` applies `accuracy_multiplier` using the selected
   local reference.
3. The JSON report records both measured accuracy and the local gate context.
4. Competition scoring continues to use the unchanged defaults in
   `inference_opt.benchmark.scoring`.

## Verification

- Unit-test the local baseline constant and report fields.
- Verify the official `0.40` scoring tests remain unchanged and passing.
- Re-evaluate an existing GPQA baseline result and confirm:
  - accuracy is `43 / 120`;
  - delta is zero;
  - multiplier is one.
- Run the complete test suite before committing implementation.

## Baseline Context

- Organizer baseline score: approximately 14.
- Local hardware: NVIDIA L4, used only for relative comparisons.
- Local GPQA baseline: 43 correct out of 120, repeated identically three times.
- Local trace baseline: three successful 120-request runs; use the median raw
  latency and throughput values as the performance reference.
