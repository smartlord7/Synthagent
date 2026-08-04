import unittest
import os
import sys
import json
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import run_batch_pipeline as rbp


class _FakeResponse:
    """Minimal stand-in for the object urllib.request.urlopen()'s context manager yields."""
    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class TestSlugifyModelId(unittest.TestCase):
    def test_slashes_and_dots_become_underscores(self):
        self.assertEqual(rbp._slugify_model_id("google/gemma-4-e2b"), "backend_google_gemma_4_e2b")

    def test_already_safe_id(self):
        self.assertEqual(rbp._slugify_model_id("qwen2.5-vl-7b-instruct"), "backend_qwen2_5_vl_7b_instruct")

    def test_lowercases(self):
        self.assertEqual(rbp._slugify_model_id("Meta-Llama-3"), "backend_meta_llama_3")


class TestDiscoverModelSweepArms(unittest.TestCase):
    """_discover_model_sweep_arms() queries the live server's GET /v1/models and turns
    every OTHER chat-capable model into a sweep arm -- this is what makes '--ablation'
    test every model actually available on the server, not a hardcoded list."""

    def setUp(self):
        self._orig_pipeline = dict(rbp.PIPELINE)

    def tearDown(self):
        rbp.PIPELINE.clear()
        rbp.PIPELINE.update(self._orig_pipeline)

    def _server_payload(self):
        return {
            "data": [
                {"id": "google/gemma-4-e2b", "object": "model"},
                {"id": "qwen2.5-vl-7b-instruct", "object": "model"},
                {"id": "text-embedding-nomic-embed-text-v1.5", "object": "model"},
            ],
            "object": "list",
        }

    def test_excludes_default_model_and_embedding_models(self):
        rbp.PIPELINE["provider"] = "local"
        rbp.PIPELINE["endpoint_url"] = "http://127.0.0.1:1234/v1/chat/completions"
        with mock.patch("urllib.request.urlopen", return_value=_FakeResponse(self._server_payload())):
            arms = rbp._discover_model_sweep_arms(exclude_model_id="google/gemma-4-e2b")
        self.assertEqual(arms, {"backend_qwen2_5_vl_7b_instruct": "qwen2.5-vl-7b-instruct"})

    def test_no_exclusion_still_drops_embedding_model(self):
        rbp.PIPELINE["provider"] = "local"
        rbp.PIPELINE["endpoint_url"] = "http://127.0.0.1:1234/v1/chat/completions"
        with mock.patch("urllib.request.urlopen", return_value=_FakeResponse(self._server_payload())):
            arms = rbp._discover_model_sweep_arms(exclude_model_id="nonexistent-model")
        self.assertEqual(set(arms.values()), {"google/gemma-4-e2b", "qwen2.5-vl-7b-instruct"})
        self.assertNotIn("text-embedding-nomic-embed-text-v1.5", arms.values())

    def test_gemini_provider_returns_empty_without_network_call(self):
        rbp.PIPELINE["provider"] = "gemini"
        with mock.patch("urllib.request.urlopen") as mock_urlopen:
            arms = rbp._discover_model_sweep_arms(exclude_model_id="gemini-2.0-flash")
        self.assertEqual(arms, {})
        mock_urlopen.assert_not_called()

    def test_unreachable_server_returns_empty_not_raise(self):
        rbp.PIPELINE["provider"] = "local"
        rbp.PIPELINE["endpoint_url"] = "http://127.0.0.1:1234/v1/chat/completions"
        with mock.patch("urllib.request.urlopen", side_effect=OSError("connection refused")):
            arms = rbp._discover_model_sweep_arms(exclude_model_id="google/gemma-4-e2b")
        self.assertEqual(arms, {})

    def test_empty_model_list_returns_empty(self):
        rbp.PIPELINE["provider"] = "local"
        rbp.PIPELINE["endpoint_url"] = "http://127.0.0.1:1234/v1/chat/completions"
        with mock.patch("urllib.request.urlopen", return_value=_FakeResponse({"data": []})):
            arms = rbp._discover_model_sweep_arms(exclude_model_id="google/gemma-4-e2b")
        self.assertEqual(arms, {})

    def test_only_default_model_loaded_returns_empty(self):
        rbp.PIPELINE["provider"] = "local"
        rbp.PIPELINE["endpoint_url"] = "http://127.0.0.1:1234/v1/chat/completions"
        payload = {"data": [{"id": "google/gemma-4-e2b", "object": "model"}]}
        with mock.patch("urllib.request.urlopen", return_value=_FakeResponse(payload)):
            arms = rbp._discover_model_sweep_arms(exclude_model_id="google/gemma-4-e2b")
        self.assertEqual(arms, {})


if __name__ == "__main__":
    unittest.main()
