# Repository Structure Design

## Goal

Create a shallow, production-lite repository layout for backend LLM inference optimization work, with clear folder ownership, explicit Python module boundaries, and a minimal personal Codex skill for fast repo navigation.

## Decisions

- Use the approved top-level layout: `configs`, `docker`, `src`, `scripts`, `evals`, `data`, `tests`, `docs`, and `results`.
- Keep root focused on operating files: `AGENTS.md`, Docker Compose entrypoints, and Python project metadata.
- Keep Python code under `src/inference_opt`.
- Keep source depth at `src/inference_opt/<domain>/<file>.py`.
- Keep `scripts` as thin command wrappers; reusable logic belongs in `src`.
- Keep generated local outputs in `results`, ignored by Git.
- Keep the personal skill named `repo-navigation`, outside the repository under the user's Codex skills directory.

## Source Modules

- `serving`: vLLM serving arguments and config validation.
- `trace`: trace JSONL schema and loading.
- `clients`: OpenAI-compatible HTTP client code.
- `benchmark`: trace replay, TTFT/TPOT capture, and ERS scoring.
- `eval`: GPQA prompting, answer extraction, and accuracy scoring.

Dependency direction:

```text
trace -> none
serving -> none
clients -> none
benchmark -> trace, clients
eval -> clients
```

## Dependencies

- Python 3.11+.
- `httpx` for HTTP and streaming client work.
- `pytest` as an optional development dependency.
- `datasets`, `huggingface-hub`, and `tqdm` as optional evaluation dependencies.
- vLLM remains the serving framework through the organizer-compatible OpenAI API server entrypoint.

## Guardrails

- Preserve `python3 -m vllm.entrypoints.openai.api_server`.
- Do not treat `Qwen3.5-2B` as a plain Transformer when changing serving behavior.
- Do not accept optimization changes without measuring trace latency and GPQA-style accuracy.
- Do not tune for `1g.18gb` without knowing the available local dev/test hardware.
