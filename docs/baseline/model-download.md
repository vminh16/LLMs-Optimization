# Model Download

Phase 1 downloads the baseline model weights without changing serving behavior.
Because the organizer revision is not available locally, this phase downloads the default Hugging Face revision and records the resolved commit hash in a manifest.

## Inputs

- Model id: `Qwen/Qwen3.5-2B`
- Requested revision: blank by default
- Local directory: `models/Qwen3.5-2B`
- Manifest: `data/manifests/model-qwen3.5-2b.json`

If the organizer later provides the pinned revision, set `MODEL_REVISION` in `.env` and run the same script again.

## Command

Activate `.venv`, then run:

```powershell
python scripts/download_model.py
```

For a no-network manifest preview:

```powershell
python scripts/download_model.py --dry-run --manifest-path data/manifests/model-qwen3.5-2b.dry-run.json
```

## Verification

After download:

```powershell
Test-Path models/Qwen3.5-2B\config.json
Test-Path models/Qwen3.5-2B\tokenizer_config.json
Get-Content data\manifests\model-qwen3.5-2b.json
```

The manifest must include `revision_resolved`.
It must not include `HF_TOKEN` or any token value.
