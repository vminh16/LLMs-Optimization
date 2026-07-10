from pathlib import Path
import unittest

from inference_opt.clients.health import build_models_url, extract_model_ids, require_model


ROOT = Path(__file__).resolve().parents[1]


class LocalComposeTest(unittest.TestCase):
    def read_experiment_compose(self, name):
        return (ROOT / "configs" / "experiments" / name).read_text(encoding="utf-8")

    def test_local_compose_mounts_default_model_without_changing_baseline_entrypoint(self):
        compose = (ROOT / "docker-compose.local.yml").read_text(encoding="utf-8")

        self.assertIn("image: vllm/vllm-openai:v0.22.1", compose)
        self.assertIn("- ./models/Qwen3.5-2B:/model:ro", compose)
        self.assertIn("- python3 #Don't change this to vllm-server", compose)
        self.assertIn("- -m  #Don't change this to vllm-server", compose)
        self.assertIn("- vllm.entrypoints.openai.api_server #Don't change this to vllm-server", compose)
        self.assertIn("- --model=/model #Don't change this to vllm-server", compose)
        self.assertIn("- --served-model-name=Qwen3.5-2B #Don't change this to vllm-server", compose)
        self.assertIn("- --enable-prefix-caching", compose)

    def test_submission_compose_stays_free_of_local_model_mounts(self):
        compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")

        self.assertNotIn("./models/Qwen3.5-2B", compose)
        self.assertNotIn("volumes:", compose)

    def test_submission_compose_uses_kv_fp8_scheduler_candidate(self):
        compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")

        self.assertIn("- python3 #Don't change this to vllm-server", compose)
        self.assertIn("- -m  #Don't change this to vllm-server", compose)
        self.assertIn("- vllm.entrypoints.openai.api_server #Don't change this to vllm-server", compose)
        self.assertIn("- --enable-prefix-caching", compose)
        self.assertIn("- --kv-cache-dtype=fp8", compose)
        self.assertIn("- --calculate-kv-scales", compose)
        self.assertIn("- --max-num-seqs=64", compose)
        self.assertNotIn("--prefix-caching-hash-algo", compose)
        self.assertNotIn("--quantization=fp8", compose)

    def test_prefix_xxhash_experiment_overrides_only_model_command(self):
        compose = self.read_experiment_compose("freewin-prefix-xxhash.compose.yml")

        self.assertIn("- --model=/model #Don't change this to vllm-server", compose)
        self.assertIn("- --served-model-name=Qwen3.5-2B #Don't change this to vllm-server", compose)
        self.assertIn("- --enable-prefix-caching", compose)
        self.assertIn("- --prefix-caching-hash-algo=xxhash", compose)
        self.assertNotIn("entrypoint:", compose)
        self.assertNotIn("image:", compose)
        self.assertNotIn("volumes:", compose)
        self.assertNotIn("--kv-cache-dtype", compose)

    def test_kv_fp8_experiment_overrides_only_model_command(self):
        compose = self.read_experiment_compose("freewin-kv-fp8.compose.yml")

        self.assertIn("- --model=/model #Don't change this to vllm-server", compose)
        self.assertIn("- --served-model-name=Qwen3.5-2B #Don't change this to vllm-server", compose)
        self.assertIn("- --enable-prefix-caching", compose)
        self.assertIn("- --kv-cache-dtype=fp8", compose)
        self.assertIn("- --calculate-kv-scales", compose)
        self.assertNotIn("entrypoint:", compose)
        self.assertNotIn("image:", compose)
        self.assertNotIn("volumes:", compose)
        self.assertNotIn("--prefix-caching-hash-algo", compose)

    def test_build_models_url_normalizes_openai_base_url(self):
        self.assertEqual(
            build_models_url("http://localhost:8000/v1"),
            "http://localhost:8000/v1/models",
        )
        self.assertEqual(
            build_models_url("http://localhost:8000/v1/"),
            "http://localhost:8000/v1/models",
        )

    def test_require_model_accepts_openai_models_payload(self):
        payload = {"data": [{"id": "Qwen3.5-2B"}]}

        self.assertEqual(extract_model_ids(payload), ["Qwen3.5-2B"])
        require_model(payload, "Qwen3.5-2B")

    def test_require_model_rejects_missing_model(self):
        payload = {"data": [{"id": "other"}]}

        with self.assertRaisesRegex(ValueError, "Qwen3.5-2B"):
            require_model(payload, "Qwen3.5-2B")


if __name__ == "__main__":
    unittest.main()
