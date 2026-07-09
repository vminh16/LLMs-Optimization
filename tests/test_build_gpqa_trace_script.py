from unittest.mock import patch
import unittest

from scripts.build_gpqa_trace import resolve_hf_token


class BuildGPQATraceScriptTest(unittest.TestCase):
    def test_resolve_hf_token_prefers_env_file_over_process_env(self):
        with patch.dict("os.environ", {"HF_TOKEN": "hf_process"}, clear=False):
            self.assertEqual(resolve_hf_token({"HF_TOKEN": "hf_file"}), "hf_file")

    def test_resolve_hf_token_reads_process_env_when_env_file_is_empty(self):
        with patch.dict("os.environ", {"HF_TOKEN": "hf_process"}, clear=False):
            self.assertEqual(resolve_hf_token({}), "hf_process")


if __name__ == "__main__":
    unittest.main()
