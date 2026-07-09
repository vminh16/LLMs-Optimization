from pathlib import Path
import json
import tempfile
import unittest

from inference_opt.serving.model_download import (
    DownloadConfig,
    build_manifest,
    load_env_file,
    resolve_config,
    write_manifest,
)


class ModelDownloadTest(unittest.TestCase):
    def test_resolve_config_uses_default_model_without_revision(self):
        config = resolve_config({})

        self.assertEqual(config.model_id, "Qwen/Qwen3.5-2B")
        self.assertIsNone(config.revision)
        self.assertEqual(config.local_dir, Path("models/Qwen3.5-2B"))
        self.assertEqual(config.manifest_path, Path("data/manifests/model-qwen3.5-2b.json"))

    def test_load_env_file_ignores_comments_and_blank_lines(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env_file = Path(tmpdir) / ".env"
            env_file.write_text(
                "\n# local settings\nMODEL_ID=Qwen/Qwen3.5-2B\nMODEL_REVISION=\n",
                encoding="utf-8",
            )

            values = load_env_file(env_file)

        self.assertEqual(values["MODEL_ID"], "Qwen/Qwen3.5-2B")
        self.assertEqual(values["MODEL_REVISION"], "")

    def test_load_env_file_strips_optional_quotes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env_file = Path(tmpdir) / ".env"
            env_file.write_text('HF_TOKEN="hf_local"\nMODEL_ID=\'Qwen/Qwen3.5-2B\'\n', encoding="utf-8")

            values = load_env_file(env_file)

        self.assertEqual(values["HF_TOKEN"], "hf_local")
        self.assertEqual(values["MODEL_ID"], "Qwen/Qwen3.5-2B")

    def test_manifest_records_requested_default_revision_without_token(self):
        config = DownloadConfig(
            model_id="Qwen/Qwen3.5-2B",
            revision=None,
            local_dir=Path("models/Qwen3.5-2B"),
            manifest_path=Path("data/manifests/model-qwen3.5-2b.json"),
        )

        manifest = build_manifest(
            config=config,
            snapshot_path=Path("models/Qwen3.5-2B/snapshots/abc123"),
            resolved_revision="abc123",
        )

        self.assertEqual(manifest["model_id"], "Qwen/Qwen3.5-2B")
        self.assertIsNone(manifest["revision_requested"])
        self.assertEqual(manifest["revision_resolved"], "abc123")
        self.assertEqual(manifest["snapshot_path"], "models/Qwen3.5-2B/snapshots/abc123")
        self.assertNotIn("token", json.dumps(manifest).lower())

    def test_write_manifest_creates_parent_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "data" / "manifests" / "model.json"
            write_manifest({"model_id": "Qwen/Qwen3.5-2B"}, manifest_path)

            self.assertTrue(manifest_path.exists())
            self.assertEqual(
                json.loads(manifest_path.read_text(encoding="utf-8"))["model_id"],
                "Qwen/Qwen3.5-2B",
            )


if __name__ == "__main__":
    unittest.main()
