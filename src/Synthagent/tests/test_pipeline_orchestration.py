import unittest
from unittest import mock
import pandas as pd
import numpy as np
from services.pipeline_manager import (
    run_single_dataset_benchmark,
    run_smote_dataset_benchmark,
    run_stochasticity_check,
    apply_fdr_correction,
)
from services.rule_inference import RuleInference

class TestPipelineOrchestration(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "f1": np.random.rand(20),
            "f2": np.random.rand(20),
            "target": np.random.choice([0, 1], size=20)
        })
        self.col_defs = [
            {"field": "f1", "headerName": "F1"},
            {"field": "f2", "headerName": "F2"},
            {"field": "target", "headerName": "Target"}
        ]
        self.config = {
            "folds": 2,
            "ml_config": {"selected_models": ["rf"], "params": {}},
            "targets": ["target"],
            "desc_metadata": "Test Metadata",
            "algorithm": "test-mock", # This will fail real LLM calls, but we check logic around it
            "ctx_ratio": 20
        }

    def test_run_single_dataset_benchmark_structure(self):
        # We need to mock the LLM client because real calls are expensive/local
        # For this logic test, we'll see if it handles the fold splitting
        
        # This function calls generate_bulk_data inside. 
        # For a truly headless unit test, we should mock generate_bulk_data.
        
        with unittest.mock.patch('services.pipeline_manager.generate_bulk_data') as mock_gen:
            mock_gen.side_effect = lambda *args, **kwargs: {"data": [{"f1": 0.5, "f2": 0.5, "target": 1}], "stats": {}}
            
            payload = ("test_ds", "test_file.csv", self.df.to_dict('records'), self.col_defs, self.config)
            result = run_single_dataset_benchmark(payload)
            
            self.assertEqual(result["dataset_id"], "test_ds")
            self.assertEqual(len(result["results"]), 2) # 2 folds
            # Check if fold_id was stamped
            for row in result["results"][0]["synthetic_data"]:
                self.assertEqual(row["_fold_id"], 1)

class TestPipelineManagerEfficiencyAndArms(unittest.TestCase):
    """
    Covers the batch-pipeline rework: rule inference + audit hoisted to run once per
    dataset, three ML arms (original/generated/augmented), and the ablation config flags.
    """

    def setUp(self):
        rng = np.random.RandomState(0)
        self.df = pd.DataFrame({
            "f1": rng.rand(40),
            "f2": rng.rand(40),
            "target": [0, 1] * 20,
        })
        self.col_defs = [
            {"field": "f1", "headerName": "F1"},
            {"field": "f2", "headerName": "F2"},
            {"field": "target", "headerName": "Target"},
        ]
        self.config = {
            "folds": 3,
            "ml_config": {"selected_models": ["rf"], "params": {}},
            "targets": ["target"],
            "desc_metadata": "Test Metadata",
            "algorithm": "test-mock",
            "gen_size_pct": 100,
        }

    @staticmethod
    def _mock_gen(*args, **kwargs):
        num_empty = kwargs.get("num_empty", 6)
        rows = [{"f1": (i % 5) / 5.0, "f2": ((i + 1) % 5) / 5.0, "target": i % 2} for i in range(num_empty)]
        return {
            "data": rows,
            "stats": {"prompt_tokens": 10, "candidates_tokens": 5, "total_tokens": 15,
                      "rule_violations": 1, "repair_attempts": 0},
        }

    def _run(self, config_overrides=None):
        cfg = dict(self.config)
        if config_overrides:
            cfg.update(config_overrides)
        payload = ("test_ds", "test_file.csv", self.df.to_dict('records'), self.col_defs, cfg)
        with mock.patch('services.pipeline_manager.generate_bulk_data', side_effect=self._mock_gen) as mock_gen:
            result = run_single_dataset_benchmark(payload)
        return result, mock_gen

    def test_rule_inference_called_once_per_dataset(self):
        with mock.patch('services.pipeline_manager.RuleInference.infer_dataset_rules',
                         wraps=RuleInference.infer_dataset_rules) as spy:
            self._run()
            self.assertEqual(spy.call_count, 1)

    def test_dataset_audit_present_once_not_per_fold(self):
        result, _ = self._run()
        self.assertIn("dataset_audit", result)
        self.assertIsInstance(result["dataset_audit"], dict)
        self.assertIn("structural", result["dataset_audit"])
        for fold in result["results"]:
            self.assertNotIn("dataset_audit", fold)

    def test_three_ml_arms_present(self):
        result, _ = self._run()
        self.assertTrue(result["results"])
        ml = result["results"][0]["ml_eval"]["rf"]
        self.assertIn("original", ml)
        self.assertIn("generated", ml)
        self.assertIn("augmented", ml)

        overall = result["overall_ml"].get("rf", {})
        if "accuracy" in overall:
            self.assertIn("gen", overall["accuracy"])
            self.assertIn("delta_gen", overall["accuracy"])
            self.assertIn("pvalue_aug_vs_real", overall["accuracy"])

    def test_inject_inferred_rules_false_empties_rule_hierarchy(self):
        _, mock_gen = self._run(config_overrides={"inject_inferred_rules": False})
        _, kwargs = mock_gen.call_args
        rules_metadata = kwargs.get("rules_metadata")
        self.assertEqual(rules_metadata["integrity"], [])
        self.assertEqual(rules_metadata["boundaries"], [])

    def test_inject_inferred_rules_true_populates_rule_hierarchy(self):
        _, mock_gen = self._run()
        _, kwargs = mock_gen.call_args
        rules_metadata = kwargs.get("rules_metadata")
        self.assertTrue(len(rules_metadata["integrity"]) > 0)

    def test_apply_post_processing_false_skips_postprocessor(self):
        cfg = dict(self.config)
        cfg["apply_post_processing"] = False
        payload = ("test_ds", "test_file.csv", self.df.to_dict('records'), self.col_defs, cfg)
        with mock.patch('services.pipeline_manager.generate_bulk_data', side_effect=self._mock_gen), \
             mock.patch('services.post_processor.PostProcessor.apply_active_rules') as mock_pp:
            run_single_dataset_benchmark(payload)
            mock_pp.assert_not_called()

    def test_apply_post_processing_true_calls_postprocessor(self):
        payload = ("test_ds", "test_file.csv", self.df.to_dict('records'), self.col_defs, self.config)
        with mock.patch('services.pipeline_manager.generate_bulk_data', side_effect=self._mock_gen), \
             mock.patch('services.post_processor.PostProcessor.apply_active_rules', side_effect=lambda data, rules: data) as mock_pp:
            run_single_dataset_benchmark(payload)
            self.assertTrue(mock_pp.called)

    def test_generation_stats_captured_per_fold(self):
        result, _ = self._run()
        for fold in result["results"]:
            self.assertIn("generation_stats", fold)
            self.assertIn("total_tokens", fold["generation_stats"])
            self.assertIn("generation_seconds", fold["generation_stats"])

    def test_fold_seconds_tracked_per_fold(self):
        """Every fold's total wall-clock time (data split + generation + all metrics/ML
        arms) must be tracked, not just LLM generation time."""
        result, _ = self._run()
        for fold in result["results"]:
            self.assertIn("fold_seconds", fold)
            self.assertIsInstance(fold["fold_seconds"], float)
            self.assertGreaterEqual(fold["fold_seconds"], 0)

    def test_c2st_and_mia_present_when_computable(self):
        """This fixture's fold sizes (well above the C2ST/MIA row-count floors) should
        always yield a computable value, not None. Uses headerName == field (unlike the
        shared self.col_defs, which renames target -> Target) so the mock LLM's lowercase
        column names actually match the real data's columns -- otherwise C2ST/MIA see zero
        common columns and correctly return None, which isn't what this test is checking."""
        col_defs = [
            {"field": "f1", "headerName": "f1"},
            {"field": "f2", "headerName": "f2"},
            {"field": "target", "headerName": "target"},
        ]
        payload = ("test_ds", "test_file.csv", self.df.to_dict('records'), col_defs, self.config)
        with mock.patch('services.pipeline_manager.generate_bulk_data', side_effect=self._mock_gen):
            result = run_single_dataset_benchmark(payload)
        for fold in result["results"]:
            self.assertIn("Gen C2ST Accuracy", fold)
            self.assertIsInstance(fold["Gen C2ST Accuracy"], float)
            self.assertIn("Gen MIA Attack AUC", fold)
            self.assertIsInstance(fold["Gen MIA Attack AUC"], float)

    def test_fidelity_by_class_present_for_imbalanced_classification(self):
        """Per-class (minority/majority) fidelity should appear on the generated-only
        corpus whenever the target is classification-like, isolating whether synthetic
        fidelity holds up on the minority class specifically."""
        rng = np.random.RandomState(1)
        imbalanced_df = pd.DataFrame({
            "f1": rng.rand(60),
            "f2": rng.rand(60),
            "target": [0] * 45 + [1] * 15,
        })
        # Use headerName == field (no rename) so config["targets"]=["target"] actually
        # matches the post-rename column name -- unlike the shared self.col_defs fixture,
        # this test checks label correctness, not just key presence.
        col_defs = [
            {"field": "f1", "headerName": "f1"},
            {"field": "f2", "headerName": "f2"},
            {"field": "target", "headerName": "target"},
        ]
        payload = ("test_ds", "test_file.csv", imbalanced_df.to_dict('records'), col_defs, self.config)
        with mock.patch('services.pipeline_manager.generate_bulk_data', side_effect=self._mock_gen):
            result = run_single_dataset_benchmark(payload)

        self.assertTrue(result["results"])
        fbc = result["results"][0].get("fidelity_by_class")
        self.assertIsNotNone(fbc)
        self.assertEqual(fbc.get("minority_label"), "1")
        self.assertEqual(fbc.get("majority_label"), "0")
        self.assertIn("minority", fbc)
        self.assertIn("majority", fbc)


class TestRunSmoteDatasetBenchmark(unittest.TestCase):
    """SMOTE baseline benchmark: pure sklearn/imblearn, deterministic via random_state --
    no LLM mocking needed. Must produce the same result shape as the LLM path so it slots
    into the same ablation-arm dispatch/aggregation code."""

    def setUp(self):
        rng = np.random.RandomState(3)
        self.df = pd.DataFrame({
            "f1": rng.rand(50),
            "f2": rng.rand(50),
            "target": [0] * 38 + [1] * 12,
        })
        self.col_defs = [
            {"field": "f1", "headerName": "f1"},
            {"field": "f2", "headerName": "f2"},
            {"field": "target", "headerName": "target"},
        ]
        self.config = {
            "folds": 2,
            "ml_config": {"selected_models": ["rf"], "params": {}},
            "targets": ["target"],
            "na_strategy": "impute",
        }

    def _run(self):
        payload = ("test_ds", "test_file.csv", self.df.to_dict('records'), self.col_defs, self.config)
        return run_smote_dataset_benchmark(payload)

    def test_result_shape_matches_llm_path(self):
        result = self._run()
        self.assertEqual(result["dataset_id"], "test_ds")
        self.assertEqual(len(result["results"]), 2)
        self.assertIn("overall_ml", result)
        self.assertIn("dataset_audit", result)
        self.assertIn("structural", result["dataset_audit"])

    def test_generation_is_near_zero_cost(self):
        result = self._run()
        for fold in result["results"]:
            stats = fold["generation_stats"]
            self.assertEqual(stats["prompt_tokens"], 0)
            self.assertEqual(stats["total_tokens"], 0)
            self.assertIn("generation_seconds", stats)

    def test_fold_seconds_tracked(self):
        result = self._run()
        for fold in result["results"]:
            self.assertIn("fold_seconds", fold)
            self.assertIsInstance(fold["fold_seconds"], float)

    def test_synthetic_rows_are_minority_class_only(self):
        """SMOTE only ever synthesizes minority-class rows -- unlike the LLM arm's
        'generated' output, which can contain any class."""
        result = self._run()
        for fold in result["results"]:
            synth_rows = fold["synthetic_data"]
            if synth_rows:
                self.assertTrue(all(row["target"] == 1 for row in synth_rows))

    def test_orig_vs_aug_comparison_populates_despite_single_class_generated_arm(self):
        """Regression test for the aggregation bug: SMOTE's 'generated' arm is inherently
        single-class, so MLEvaluator correctly refuses to score it -- that must not blank
        out the still-valid original-vs-augmented comparison."""
        result = self._run()
        overall = result["overall_ml"].get("rf", {})
        self.assertIn("accuracy", overall)
        self.assertIn("real", overall["accuracy"])
        self.assertIn("aug", overall["accuracy"])
        self.assertIn("delta", overall["accuracy"])
        self.assertNotIn("gen", overall["accuracy"])


class TestRunStochasticityCheck(unittest.TestCase):
    """Generation-stochasticity decomposition: ONE fixed train/test split and ONE fixed
    context sample, with `repeats` fresh LLM generations on those identical inputs --
    isolates pure LLM sampling noise from data-split/context-sampling variance."""

    def setUp(self):
        rng = np.random.RandomState(2)
        self.df = pd.DataFrame({
            "f1": rng.rand(40),
            "f2": rng.rand(40),
            "target": [0, 1] * 20,
        })
        self.col_defs = [
            {"field": "f1", "headerName": "f1"},
            {"field": "f2", "headerName": "f2"},
            {"field": "target", "headerName": "target"},
        ]
        self.config = {
            "ml_config": {"selected_models": ["rf"], "params": {}},
            "targets": ["target"],
            "algorithm": "test-mock",
            "ctx_size_pct": 20,
            "gen_size_pct": 100,
        }

    @staticmethod
    def _mock_gen(*args, **kwargs):
        num_empty = kwargs.get("num_empty", 6)
        rows = [{"f1": (i % 5) / 5.0, "f2": ((i + 1) % 5) / 5.0, "target": i % 2} for i in range(num_empty)]
        return {
            "data": rows,
            "stats": {"prompt_tokens": 10, "candidates_tokens": 5, "total_tokens": 15,
                      "rule_violations": 0, "repair_attempts": 0},
        }

    def _payload(self):
        return ("test_ds", "test_file.csv", self.df.to_dict('records'), self.col_defs, self.config)

    def test_returns_requested_repeat_count(self):
        with mock.patch('services.pipeline_manager.generate_bulk_data', side_effect=self._mock_gen):
            result = run_stochasticity_check(self._payload(), repeats=3)
        self.assertEqual(len(result["repeat_results"]), 3)
        for i, r in enumerate(result["repeat_results"], start=1):
            self.assertEqual(r["repeat"], i)

    def test_context_and_split_computed_exactly_once(self):
        from services.data_manager import select_context_rows
        with mock.patch('services.pipeline_manager.generate_bulk_data', side_effect=self._mock_gen), \
             mock.patch('services.pipeline_manager.select_context_rows', wraps=select_context_rows) as spy_ctx:
            run_stochasticity_check(self._payload(), repeats=3)
        self.assertEqual(spy_ctx.call_count, 1)

    def test_fold_seconds_tracked_per_repeat(self):
        with mock.patch('services.pipeline_manager.generate_bulk_data', side_effect=self._mock_gen):
            result = run_stochasticity_check(self._payload(), repeats=2)
        for r in result["repeat_results"]:
            self.assertIn("fold_seconds", r)
            self.assertIsInstance(r["fold_seconds"], float)

    def test_split_sizes_reported_and_consistent_with_dataset(self):
        with mock.patch('services.pipeline_manager.generate_bulk_data', side_effect=self._mock_gen):
            result = run_stochasticity_check(self._payload(), repeats=2)
        sizes = result["split_sizes"]
        self.assertEqual(sizes["train"] + sizes["test"], len(self.df))


class TestApplyFdrCorrection(unittest.TestCase):
    """Unit tests for the Benjamini-Hochberg correction helper used across the batch
    pipeline's and ablation study's significance tests."""

    def test_corrects_multiple_pvalues(self):
        raw = {"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.5}
        out = {}
        apply_fdr_correction(list(raw.keys()), get_fn=raw.get, set_fn=out.__setitem__)
        self.assertEqual(set(out.keys()), set(raw.keys()))
        for q in out.values():
            self.assertTrue(0.0 <= q <= 1.0)

    def test_skips_none_pvalues(self):
        raw = {"a": 0.01, "b": None, "c": 0.2}
        out = {}
        apply_fdr_correction(list(raw.keys()), get_fn=raw.get, set_fn=out.__setitem__)
        self.assertIn("a", out)
        self.assertIn("c", out)
        self.assertNotIn("b", out)

    def test_empty_refs_is_a_noop(self):
        out = {}
        apply_fdr_correction([], get_fn=lambda ref: None, set_fn=out.__setitem__)
        self.assertEqual(out, {})

    def test_all_none_is_a_noop(self):
        out = {}
        apply_fdr_correction(["a", "b"], get_fn=lambda ref: None, set_fn=out.__setitem__)
        self.assertEqual(out, {})


if __name__ == '__main__':
    unittest.main()
