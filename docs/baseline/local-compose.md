# Baseline Local Compose

Use `docker-compose.local.yml` only for local development.
It mounts the locally downloaded model at `./models/Qwen3.5-2B` into the container path `/model`.

Do not submit `docker-compose.local.yml` to the organizer portal.
Local filesystem paths such as `./models/Qwen3.5-2B` exist only on this machine.

## Submission Compose

For an untouched baseline submission, use `docker-compose.yml`.
It keeps the organizer baseline image and command shape without local volumes.

`docker-compose.yml` still points at the public baseline image:

```text
vllm/vllm-openai:v0.22.1
```

A custom Docker image is not required until the submission needs files or packages that are not already in the public vLLM image.
Examples that would require a custom image include a patched vLLM build, custom kernels, additional runtime dependencies, or pre-baked model artifacts.

## Local Startup

From the repository root:

```powershell
docker compose -f docker-compose.local.yml up model
```

In another terminal with `.venv` active:

```powershell
python scripts/check_server_health.py
```

The health check calls:

```text
GET http://localhost:8000/v1/models
```

The next phase will add the benchmark client that reads `data/trace-round1.jsonl` and sends OpenAI-compatible requests to:

```text
POST http://localhost:8000/v1/chat/completions
```
