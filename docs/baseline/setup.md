# Baseline Setup

Phase 0 creates an isolated Python environment and records the baseline tooling choices.
It does not download the model, download GPQA, start Docker, or run optimization experiments.

## Environment

Use Python 3.11 or newer.

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,eval]"
```

If `py -3.11` is unavailable, use an explicit Python executable that reports version 3.11 or newer.

Linux shell:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,eval]"
```

## Local Settings

Create `.env` from `.env.example` and fill secrets locally only.
Never commit `.env`, model files, or gated GPQA content.

Required values:

- `HF_TOKEN`: Hugging Face token with accepted access for the model and GPQA.
- `MODEL_ID`: fixed at `Qwen/Qwen3.5-2B`.
- `MODEL_REVISION`: organizer-provided pinned revision when available.
- `GPQA_DATASET_ID`: fixed at `Idavidrein/gpqa`.
- `GPQA_VARIANT`: fixed internal label `diamond`.
- `VLLM_BASE_URL`: local OpenAI-compatible vLLM base URL.

## Frameworks And Libraries

- Serving framework: `vllm/vllm-openai:v0.22.1`.
- Python package/runtime: Python 3.11+, `setuptools`, `src/` package layout.
- HTTP client: `httpx`.
- Test runner: `pytest`, with Phase 0 checks also runnable through stdlib `unittest`.
- Evaluation data tooling: `datasets`, `huggingface-hub`, `tqdm`.

## Verification

Run these from the repository root after activating `.venv`:

```powershell
python -c "import sys; print(sys.prefix)"
python -m unittest discover -s tests
```

The printed prefix should end in `.venv`.
