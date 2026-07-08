from pathlib import Path
import unittest

from inference_opt.clients.health import build_models_url, extract_model_ids, require_model


ROOT = Path(__file__).resolve().parents[1]


class LocalComposeTest(unittest.TestCase):
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
