import unittest
import os
import sys
import json
import shutil
import tempfile
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import run_batch_pipeline as rbp


class TestJsonSafe(unittest.TestCase):
    """_json_safe must convert numpy scalars/arrays to native Python types so checkpoint
    values round-trip through json.dump/json.load as real numbers, not strings -- a later
    resume feeds them straight into Wilcoxon delta math, which needs actual floats."""

    def test_converts_numpy_scalar(self):
        out = rbp._json_safe({"a": np.float64(1.5), "b": np.int64(3)})
        self.assertEqual(out, {"a": 1.5, "b": 3})
        self.assertIsInstance(out["a"], float)
        self.assertIsInstance(out["b"], int)

    def test_converts_numpy_array_in_list(self):
        out = rbp._json_safe({"vals": [np.float64(1.0), np.float64(2.0)]})
        self.assertEqual(out["vals"], [1.0, 2.0])

    def test_leaves_plain_python_values_untouched(self):
        out = rbp._json_safe({"a": 1, "b": "x", "c": [1, 2, 3], "d": None})
        self.assertEqual(out, {"a": 1, "b": "x", "c": [1, 2, 3], "d": None})

    def test_is_actually_json_serialisable(self):
        payload = rbp._json_safe({"delta": np.float64(-0.05), "series": [np.float64(0.1), np.float64(0.2)]})
        # Would raise TypeError if any numpy scalar survived.
        json.dumps(payload)


class TestCheckpointPersistence(unittest.TestCase):
    def setUp(self):
        self.run_dir = tempfile.mkdtemp(prefix="tabullm_checkpoint_test_")

    def tearDown(self):
        shutil.rmtree(self.run_dir, ignore_errors=True)

    def test_load_checkpoint_missing_file_returns_empty(self):
        cp = rbp._load_checkpoint(self.run_dir)
        self.assertEqual(cp, {"completed": {}, "in_progress": {}})

    def test_save_then_load_round_trips(self):
        cp = rbp._load_checkpoint(self.run_dir)
        rbp._mark_unit_done(self.run_dir, cp, "full::ds.csv",
                             {"dataset": "ds.csv", "arm": "full"}, {"acc": [0.1, 0.2]}, {"folds": 3})
        reloaded = rbp._load_checkpoint(self.run_dir)
        self.assertIn("full::ds.csv", reloaded["completed"])
        entry = reloaded["completed"]["full::ds.csv"]
        self.assertEqual(entry["summary_row"], {"dataset": "ds.csv", "arm": "full"})
        self.assertEqual(entry["per_fold_series"], {"acc": [0.1, 0.2]})
        self.assertEqual(entry["unit_config"], {"folds": 3})

    def test_save_checkpoint_leaves_no_leftover_tmp_file(self):
        cp = rbp._load_checkpoint(self.run_dir)
        rbp._mark_unit_done(self.run_dir, cp, "k", {"a": 1}, None, {"folds": 1})
        self.assertFalse(os.path.exists(rbp._checkpoint_path(self.run_dir) + ".tmp"))
        self.assertTrue(os.path.exists(rbp._checkpoint_path(self.run_dir)))

    def test_load_checkpoint_corrupt_file_returns_empty_not_raise(self):
        with open(rbp._checkpoint_path(self.run_dir), "w", encoding="utf-8") as fh:
            fh.write("{not valid json")
        cp = rbp._load_checkpoint(self.run_dir)
        self.assertEqual(cp, {"completed": {}, "in_progress": {}})

    def test_per_fold_series_defaults_to_empty_dict_when_none(self):
        cp = rbp._load_checkpoint(self.run_dir)
        rbp._mark_unit_done(self.run_dir, cp, "k", {"a": 1}, None, {"repeats": 5})
        self.assertEqual(cp["completed"]["k"]["per_fold_series"], {})


class TestCheckpointLookup(unittest.TestCase):
    """The seed-basis guard: a cached unit is only reusable if it was produced under the
    SAME unit_config (e.g. fold count) the caller is asking for now -- pipeline_manager
    seeds every fold purely from `42 + fold_num`, so a different fold count means a
    different, non-comparable set of splits."""

    def setUp(self):
        self.run_dir = tempfile.mkdtemp(prefix="tabullm_checkpoint_test_")
        self.checkpoint = rbp._load_checkpoint(self.run_dir)
        rbp._mark_unit_done(self.run_dir, self.checkpoint, "full::ds.csv",
                             {"dataset": "ds.csv"}, {"acc": [0.1]}, {"folds": 10})

    def tearDown(self):
        shutil.rmtree(self.run_dir, ignore_errors=True)

    def test_matching_config_returns_entry(self):
        entry = rbp._checkpoint_lookup(self.checkpoint, "full::ds.csv", {"folds": 10})
        self.assertIsNotNone(entry)
        self.assertEqual(entry["summary_row"], {"dataset": "ds.csv"})

    def test_mismatched_config_returns_none(self):
        entry = rbp._checkpoint_lookup(self.checkpoint, "full::ds.csv", {"folds": 5})
        self.assertIsNone(entry)

    def test_missing_key_returns_none(self):
        entry = rbp._checkpoint_lookup(self.checkpoint, "nonexistent::ds.csv", {"folds": 10})
        self.assertIsNone(entry)


class TestFoldLevelResume(unittest.TestCase):
    """Fold-level checkpointing: each fold is saved the instant it finishes (not just when
    the whole unit does), so a crash mid-unit only loses the one fold in flight."""

    def setUp(self):
        self.run_dir = tempfile.mkdtemp(prefix="tabullm_checkpoint_test_")
        self.checkpoint = rbp._load_checkpoint(self.run_dir)

    def tearDown(self):
        shutil.rmtree(self.run_dir, ignore_errors=True)

    def test_no_cached_folds_returns_empty_dict(self):
        cached = rbp._resume_fold_results(self.checkpoint, "full::ds.csv", {"folds": 3})
        self.assertEqual(cached, {})

    def test_saved_folds_are_returned_with_int_keys(self):
        rbp._save_fold_result(self.run_dir, self.checkpoint, "full::ds.csv", {"folds": 3}, 1, {"acc": 0.9})
        rbp._save_fold_result(self.run_dir, self.checkpoint, "full::ds.csv", {"folds": 3}, 2, {"acc": 0.8})
        cached = rbp._resume_fold_results(self.checkpoint, "full::ds.csv", {"folds": 3})
        self.assertEqual(set(cached.keys()), {1, 2})
        self.assertEqual(cached[1], {"acc": 0.9})

    def test_survives_a_reload_from_disk(self):
        rbp._save_fold_result(self.run_dir, self.checkpoint, "full::ds.csv", {"folds": 3}, 1, {"acc": 0.9})
        reloaded = rbp._load_checkpoint(self.run_dir)
        cached = rbp._resume_fold_results(reloaded, "full::ds.csv", {"folds": 3})
        self.assertEqual(cached, {1: {"acc": 0.9}})

    def test_mismatched_unit_config_is_not_reused(self):
        rbp._save_fold_result(self.run_dir, self.checkpoint, "full::ds.csv", {"folds": 3}, 1, {"acc": 0.9})
        cached = rbp._resume_fold_results(self.checkpoint, "full::ds.csv", {"folds": 10})
        self.assertEqual(cached, {}, "a different fold count means different/incomparable seeds")

    def test_mark_unit_done_clears_in_progress_cache(self):
        rbp._save_fold_result(self.run_dir, self.checkpoint, "full::ds.csv", {"folds": 3}, 1, {"acc": 0.9})
        self.assertIn("full::ds.csv", self.checkpoint["in_progress"])
        rbp._mark_unit_done(self.run_dir, self.checkpoint, "full::ds.csv",
                             {"dataset": "ds.csv"}, {}, {"folds": 3})
        self.assertNotIn("full::ds.csv", self.checkpoint["in_progress"])
        self.assertIn("full::ds.csv", self.checkpoint["completed"])


class TestFindResumableRunDir(unittest.TestCase):
    def setUp(self):
        self.results_dir = tempfile.mkdtemp(prefix="tabullm_results_test_")
        self._orig_results_dir = rbp.RESULTS_DIR
        rbp.RESULTS_DIR = self.results_dir

    def tearDown(self):
        rbp.RESULTS_DIR = self._orig_results_dir
        shutil.rmtree(self.results_dir, ignore_errors=True)

    def test_no_candidates_returns_none(self):
        self.assertIsNone(rbp._find_resumable_run_dir("ablation"))

    def test_ignores_dirs_without_checkpoint(self):
        os.makedirs(os.path.join(self.results_dir, "ablation_20260101_000000"))
        self.assertIsNone(rbp._find_resumable_run_dir("ablation"))

    def test_finds_dir_with_checkpoint(self):
        d = os.path.join(self.results_dir, "ablation_20260101_000000")
        os.makedirs(d)
        with open(rbp._checkpoint_path(d), "w", encoding="utf-8") as fh:
            json.dump({"completed": {}}, fh)
        self.assertEqual(rbp._find_resumable_run_dir("ablation"), d)

    def test_picks_most_recent_of_several(self):
        for ts in ("20260101_000000", "20260102_000000", "20260103_000000"):
            d = os.path.join(self.results_dir, f"ablation_{ts}")
            os.makedirs(d)
            with open(rbp._checkpoint_path(d), "w", encoding="utf-8") as fh:
                json.dump({"completed": {}}, fh)
        picked = rbp._find_resumable_run_dir("ablation")
        self.assertTrue(picked.endswith("ablation_20260103_000000"))

    def test_ignores_dirs_of_a_different_prefix(self):
        d = os.path.join(self.results_dir, "run_20260101_000000")
        os.makedirs(d)
        with open(rbp._checkpoint_path(d), "w", encoding="utf-8") as fh:
            json.dump({"completed": {}}, fh)
        self.assertIsNone(rbp._find_resumable_run_dir("ablation"))


class TestResolveRunDir(unittest.TestCase):
    def setUp(self):
        self.results_dir = tempfile.mkdtemp(prefix="tabullm_results_test_")
        self._orig_results_dir = rbp.RESULTS_DIR
        rbp.RESULTS_DIR = self.results_dir

    def tearDown(self):
        rbp.RESULTS_DIR = self._orig_results_dir
        shutil.rmtree(self.results_dir, ignore_errors=True)

    def test_no_resume_always_starts_fresh(self):
        run_dir, resumed = rbp._resolve_run_dir(None, "ablation", "20260101_000000")
        self.assertFalse(resumed)
        self.assertTrue(os.path.isdir(run_dir))
        self.assertIn("ablation_20260101_000000", run_dir)

    def test_bare_resume_with_no_candidates_falls_back_to_fresh(self):
        run_dir, resumed = rbp._resolve_run_dir("__latest__", "ablation", "20260101_000000")
        self.assertFalse(resumed)
        self.assertTrue(os.path.isdir(run_dir))

    def test_bare_resume_picks_existing_checkpoint_dir(self):
        d = os.path.join(self.results_dir, "ablation_20250101_000000")
        os.makedirs(d)
        with open(rbp._checkpoint_path(d), "w", encoding="utf-8") as fh:
            json.dump({"completed": {}}, fh)
        run_dir, resumed = rbp._resolve_run_dir("__latest__", "ablation", "20260101_000000")
        self.assertTrue(resumed)
        self.assertEqual(run_dir, d)

    def test_explicit_nonexistent_path_falls_back_to_fresh(self):
        run_dir, resumed = rbp._resolve_run_dir(
            os.path.join(self.results_dir, "does_not_exist"), "ablation", "20260101_000000")
        self.assertFalse(resumed)
        self.assertTrue(os.path.isdir(run_dir))

    def test_explicit_existing_path_resumes_it(self):
        d = os.path.join(self.results_dir, "my_custom_dir")
        os.makedirs(d)
        run_dir, resumed = rbp._resolve_run_dir(d, "ablation", "20260101_000000")
        self.assertTrue(resumed)
        self.assertEqual(run_dir, d)


if __name__ == "__main__":
    unittest.main()
