from __future__ import annotations

import argparse
import json
from pathlib import Path

from inference_opt.serving.model_download import (
    build_manifest,
    download_model,
    load_env_file,
    resolve_config,
    write_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the baseline model snapshot from Hugging Face.")
    parser.add_argument("--env-file", default=".env", help="Local env file with HF_TOKEN and model settings.")
    parser.add_argument("--model-id", default=None, help="Hugging Face model id.")
    parser.add_argument("--revision", default=None, help="Optional Hugging Face revision. Empty means default.")
    parser.add_argument("--local-dir", default=None, help="Local model directory.")
    parser.add_argument("--manifest-path", default=None, help="Manifest output path.")
    parser.add_argument("--dry-run", action="store_true", help="Write a manifest preview without downloading.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    values = load_env_file(Path(args.env_file))
    config = resolve_config(
        values,
        model_id=args.model_id,
        revision=args.revision,
        local_dir=args.local_dir,
        manifest_path=args.manifest_path,
    )

    if args.dry_run:
        manifest = build_manifest(
            config=config,
            snapshot_path=config.local_dir,
            resolved_revision="dry-run",
        )
        write_manifest(manifest, config.manifest_path)
    else:
        manifest = download_model(config)

    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
