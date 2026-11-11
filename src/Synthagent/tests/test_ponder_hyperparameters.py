import unittest
import pandas as pd
import numpy as np
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from synthagent import Synthagent


class TestPonderHyperparameters(unittest.TestCase):
    """Unit tests for Synthagent.ponder_hyperparameters(): the structured, JSON-based
    counterpart to the free-text ponder_strategy(). Never raises -- any missing/invalid
    field, or a totally unparseable response, falls back to caller-supplied defaults (or
    safe hard-coded defaults)."""

    def setUp(self):
        self.lib = Synthagent(provider="local", model="test-ponder-model")
        rng = np.random.RandomState(0)
        self.df = pd.DataFrame({
            "age": rng.randint(18, 80, 30),
            "income": rng.rand(30) * 100000,
            "target": rng.choice([0, 1], size=30),
        })

    @patch('synthagent.api.LocalOpenAIProvider.call_llm')
    def test_valid_json_response_is_used_verbatim(self, mock_call):
        mock_call.return_value = '{"ctx_size_pct": 35, "strategy": "dense", "temperature": 0.5}'
        result = self.lib.ponder_hyperparameters(self.df, targets=["target"])
        self.assertEqual(result, {"ctx_size_pct": 35, "strategy": "dense", "temperature": 0.5})

    @patch('synthagent.api.LocalOpenAIProvider.call_llm')
    def test_malformed_response_falls_back_to_defaults(self, mock_call):
        mock_call.return_value = "not valid json at all, sorry!"
        defaults = {"ctx_size_pct": 25, "strategy": "diverse", "temperature": 0.6}
        result = self.lib.ponder_hyperparameters(self.df, targets=["target"], defaults=defaults)
        self.assertEqual(result, defaults)

    @patch('synthagent.api.LocalOpenAIProvider.call_llm')
    def test_out_of_range_fields_individually_fall_back(self, mock_call):
        # ctx_size_pct out of [5, 50] range, strategy invalid enum, temperature valid.
        mock_call.return_value = '{"ctx_size_pct": 999, "strategy": "bogus", "temperature": 0.8}'
        defaults = {"ctx_size_pct": 20, "strategy": "random", "temperature": 0.7}
        result = self.lib.ponder_hyperparameters(self.df, targets=["target"], defaults=defaults)
        self.assertEqual(result["ctx_size_pct"], 20)     # fell back
        self.assertEqual(result["strategy"], "random")   # fell back
        self.assertEqual(result["temperature"], 0.8)     # kept (valid)

    @patch('synthagent.api.LocalOpenAIProvider.call_llm')
    def test_partial_response_keeps_valid_fields_and_fills_missing(self, mock_call):
        mock_call.return_value = '{"strategy": "diverse"}'
        defaults = {"ctx_size_pct": 20, "strategy": "random", "temperature": 0.7}
        result = self.lib.ponder_hyperparameters(self.df, targets=["target"], defaults=defaults)
        self.assertEqual(result["strategy"], "diverse")
        self.assertEqual(result["ctx_size_pct"], 20)
        self.assertEqual(result["temperature"], 0.7)

    @patch('synthagent.api.LocalOpenAIProvider.call_llm')
    def test_non_dict_json_falls_back_to_defaults(self, mock_call):
        mock_call.return_value = '[1, 2, 3]'
        defaults = {"ctx_size_pct": 30, "strategy": "dense", "temperature": 0.4}
        result = self.lib.ponder_hyperparameters(self.df, targets=["target"], defaults=defaults)
        self.assertEqual(result, defaults)

    @patch('synthagent.api.LocalOpenAIProvider.call_llm')
    def test_llm_exception_falls_back_to_defaults_without_raising(self, mock_call):
        mock_call.side_effect = RuntimeError("LM Studio unreachable")
        defaults = {"ctx_size_pct": 15, "strategy": "diverse", "temperature": 0.9}
        result = self.lib.ponder_hyperparameters(self.df, targets=["target"], defaults=defaults)
        self.assertEqual(result, defaults)

    @patch('synthagent.api.LocalOpenAIProvider.call_llm')
    def test_no_defaults_uses_hardcoded_safe_values(self, mock_call):
        mock_call.return_value = None
        result = self.lib.ponder_hyperparameters(self.df, targets=["target"])
        self.assertEqual(result, {"ctx_size_pct": 20, "strategy": "random", "temperature": 0.7})


if __name__ == "__main__":
    unittest.main()
