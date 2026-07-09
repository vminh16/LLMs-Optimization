from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping
import json
import os


DEFAULT_MODEL_ID = "Qwen/Qwen3.5-2B"
DEFAULT_LOCAL_DIR = Path("models/Qwen3.5-2B")
DEFAULT_MANIFEST_PATH = Path("data/manifests/model-qwen3.5-2b.json")


@dataclass(frozen=True)
class DownloadConfig:
    model_id: str
    revision: str | None
    local_dir: Path
    manifest_path: Path
    token: str | None = None


def load_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = _strip_optional_quotes(value)
    return values


def _strip_optional_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _blank_to_none(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


def resolve_config(
    values: Mapping[str, str],
    *,
    model_id: str | None = None,
    revision: str | None = None,
    local_dir: str | None = None,
    manifest_path: str | None = None,
) -> DownloadConfig:
    resolved_model_id = model_id or values.get("MODEL_ID") or DEFAULT_MODEL_ID
    resolved_revision = _blank_to_none(revision if revision is not None else values.get("MODEL_REVISION"))
    resolved_local_dir = Path(local_dir or values.get("MODEL_LOCAL_DIR") or DEFAULT_LOCAL_DIR)
    resolved_manifest_path = Path(manifest_path or values.get("MODEL_MANIFEST_PATH") or DEFAULT_MANIFEST_PATH)
    token = _blank_to_none(values.get("HF_TOKEN") or os.environ.get("HF_TOKEN"))

    return DownloadConfig(
        model_id=resolved_model_id,
        revision=resolved_revision,
        local_dir=resolved_local_dir,
        manifest_path=resolved_manifest_path,
        token=token,
    )


def build_manifest(
    *,
    config: DownloadConfig,
    snapshot_path: Path,
    resolved_revision: str,
) -> dict[str, str | None]:
    return {
        "model_id": config.model_id,
        "revision_requested": config.revision,
        "revision_resolved": resolved_revision,
        "snapshot_path": snapshot_path.as_posix(),
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
    }


def write_manifest(manifest: Mapping[str, str | None], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def download_model(config: DownloadConfig) -> dict[str, str | None]:
    from huggingface_hub import HfApi, snapshot_download

    info = HfApi().model_info(
        repo_id=config.model_id,
        revision=config.revision,
        token=config.token,
    )
    snapshot_path = Path(
        snapshot_download(
            repo_id=config.model_id,
            revision=config.revision,
            token=config.token,
            local_dir=config.local_dir,
        )
    )
    manifest = build_manifest(
        config=config,
        snapshot_path=snapshot_path,
        resolved_revision=info.sha,
    )
    write_manifest(manifest, config.manifest_path)
    return manifest
