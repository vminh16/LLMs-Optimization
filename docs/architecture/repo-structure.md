# Repository Structure

This repository uses a production-lite layout for inference optimization work.
Keep the tree shallow, with each folder owning one clear responsibility.

```text
.
├── configs/        # Serving and benchmark configuration files.
├── docker/         # Dockerfile and image build assets when needed.
├── src/            # Reusable Python package code.
├── scripts/        # Thin CLI wrappers around src modules.
├── evals/          # Evaluation entrypoints and GPQA harness assets.
├── data/           # Input traces and small dataset metadata.
├── tests/          # Pytest tests for src modules and scripts.
├── docs/           # Architecture notes, specs, plans, and decisions.
└── results/        # Local benchmark/eval outputs; ignored by Git.
```

## Source Boundaries

```text
src/
└── inference_opt/
    ├── serving/    # vLLM serving arguments and config validation.
    ├── trace/      # trace-round1 JSONL schema and loading.
    ├── clients/    # OpenAI-compatible HTTP client code.
    ├── benchmark/  # trace replay, TTFT/TPOT capture, ERS scoring.
    └── eval/       # GPQA prompting, answer extraction, accuracy scoring.
```

Dependency direction:

- `serving` does not import `trace`, `clients`, `benchmark`, or `eval`.
- `trace` does not import other project modules.
- `clients` does not import `benchmark` or `eval`.
- `benchmark` may import `trace` and `clients`.
- `eval` may import `clients`.

Avoid generic `utils` modules until two concrete call sites need the same helper.
