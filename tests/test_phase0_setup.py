from pathlib import Path
import tomllib
import unittest


ROOT = Path(__file__).resolve().parents[1]


class Phase0SetupTest(unittest.TestCase):
    def test_env_example_declares_safe_local_baseline_settings(self):
        env_example = ROOT / ".env.example"

        self.assertTrue(env_example.exists())
        values = {}
        for line in env_example.read_text(encoding="utf-8").splitlines():
            if not line or line.startswith("#"):
                continue
            key, value = line.split("=", 1)
            values[key] = value

        self.assertEqual(values["MODEL_ID"], "Qwen/Qwen3.5-2B")
        self.assertEqual(values["GPQA_DATASET_ID"], "Idavidrein/gpqa")
        self.assertEqual(values["GPQA_VARIANT"], "diamond")
        self.assertEqual(values["VLLM_BASE_URL"], "http://localhost:8000/v1")
        self.assertIn("HF_TOKEN", values)
        self.assertEqual(values["HF_TOKEN"], "")
        self.assertEqual(values["MODEL_REVISION"], "")

    def test_pyproject_declares_phase0_dependencies(self):
        pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        dependencies = pyproject["project"]["dependencies"]
        optional = pyproject["project"]["optional-dependencies"]

        self.assertTrue(any(dep.startswith("httpx") for dep in dependencies))
        self.assertTrue(any(dep.startswith("pytest") for dep in optional["dev"]))
        self.assertTrue(any(dep.startswith("datasets") for dep in optional["eval"]))
        self.assertTrue(any(dep.startswith("huggingface-hub") for dep in optional["eval"]))
        self.assertTrue(any(dep.startswith("tqdm") for dep in optional["eval"]))

    def test_gitignore_keeps_local_secrets_models_and_gated_gpqa_out_of_git(self):
        gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()

        self.assertIn(".env", gitignore)
        self.assertIn(".venv/", gitignore)
        self.assertIn("models/", gitignore)
        self.assertIn("data/gpqa/", gitignore)
        self.assertIn("data/traces/gpqa-*.jsonl", gitignore)

    def test_baseline_setup_doc_exists_and_names_phase0_verification_commands(self):
        setup_doc = ROOT / "docs" / "baseline" / "setup.md"

        self.assertTrue(setup_doc.exists())
        content = setup_doc.read_text(encoding="utf-8")
        self.assertIn("Phase 0", content)
        self.assertIn(".venv", content)
        self.assertIn("python -m unittest discover -s tests", content)


if __name__ == "__main__":
    unittest.main()
