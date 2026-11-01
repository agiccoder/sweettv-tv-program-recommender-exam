"""Synthetic full CLI and frozen replacement-rule tests; not real quality trials."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from sweettv.data import build_interactions, import_data, sha256
from sweettv.improve_v3 import run, select_strategy
from sweettv.v3_models import Family
from test_pipeline_data import build_fixture, HUGE_USER, COLD_USER

ROOT = Path(__file__).resolve().parents[1]


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "config/improvement-v3.json").read_text())
        self.config["methods"] = [spec for spec in self.config["methods"] if spec["name"] in {"ppr_v2", "channel_time"}]

    def summaries(self, values):
        return {block["name"]: {"methods": {name: {"metrics": {
            kind: {"map_at_5": scores[i]} for kind in ["primary", "secondary"]}}
            for name, scores in values.items()}} for i, block in enumerate(self.config["slices"])}

    def test_mean_gain_needs_two_strict_slice_wins_and_exact_ties_prefer_v2(self):
        one_win = self.summaries({"ppr_v2": [.32] * 3, "channel_time": [.6, .3, .3]})
        result = select_strategy(one_win, self.config)
        self.assertEqual(result["raw_best_method"], "channel_time")
        self.assertEqual(result["selected_method"], "ppr_v2")
        self.assertFalse(result["recommended_replacement_of_v2"])
        two_wins = self.summaries({"ppr_v2": [.32] * 3, "channel_time": [.35, .35, .3]})
        self.assertEqual(select_strategy(two_wins, self.config)["selected_method"], "channel_time")
        ties = self.summaries({"ppr_v2": [.32] * 3, "channel_time": [.32] * 3})
        self.assertEqual(select_strategy(ties, self.config)["selected_method"], "ppr_v2")

    def test_both_proxies_and_only_complete_methods_enter_selection(self):
        summaries = self.summaries({"ppr_v2": [.32] * 3, "channel_time": [.30] * 3})
        for result in summaries.values():
            result["methods"]["channel_time"]["metrics"]["secondary"]["map_at_5"] = .40
        self.assertEqual(select_strategy(summaries, self.config)["selected_method"], "channel_time")
        del summaries["weeks12"]["methods"]["channel_time"]
        selected = select_strategy(summaries, self.config)
        self.assertEqual(selected["selected_method"], "ppr_v2")
        self.assertEqual(selected["excluded_incomplete_methods"], ["channel_time"])

    def test_native_retrieval_unions_have_frozen_250_and_300_bounds(self):
        family = object.__new__(Family)
        family.config = json.loads((ROOT / "config/improvement-v3.json").read_text())
        family.pool = set(range(1, 401))
        names = ["ppr_v2", "history_seconds", "history_watches", "history_recent", "channel_time", "als32"]
        lists = {name: [(item, float(1000 - item)) for item in range(i * 50 + 1, i * 50 + 51)] for i, name in enumerate(names)}
        ordinary = next(spec for spec in family.config["methods"] if spec["name"] == "hybrid_50_25_25")
        scores, contributions, union = family.compose(ordinary, lists)
        self.assertEqual(len(union), 250)
        self.assertEqual(set(scores), union)
        als = next(spec for spec in family.config["methods"] if spec["name"] == "hybrid_als_ppr")
        scores, contributions, union = family.compose(als, lists)
        self.assertEqual(len(union), 300)
        self.assertEqual(set(scores), union)
        self.assertEqual(contributions[1]["ppr_v2"]["weight"], .25)
        self.assertEqual(contributions[251]["als32"]["weight"], .25)


class RunnerTests(unittest.TestCase):
    def test_all16_on_three_synthetic_slices_refit_roundtrip_and_actual_exports(self):
        spec = importlib.util.spec_from_file_location("prepare_v3_integration", ROOT / "scripts/prepare_improvement_v3.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base, raw = root / "base", root / "raw"
            base.mkdir()
            build_fixture(raw)
            audit = import_data(raw, base / "raw.sqlite", base / "audit")
            build_interactions(base / "raw.sqlite", audit, base / "prepared")
            config = json.loads((ROOT / "config/improvement-v3.json").read_text())
            for block in config["slices"]:
                block.update({"train_start": "2020-01-01T00:00:00", "train_end": "2020-01-02T00:00:00", "target_end": "2020-01-03T00:00:00"})
            config["final"] = {"train_start": "2020-01-01T00:00:00", "train_end": "2020-01-03T00:00:00", "target_end": "2020-01-04T00:00:00"}
            config["examples"] = [HUGE_USER, COLD_USER]
            config_path = root / "fixture-config.json"
            config_path.write_text(json.dumps(config))
            prepared = root / "prepared"
            module.prepare(base, prepared, config_path, raw / "submission.csv", expected_users=3)
            before = sha256(base / "raw.sqlite")
            args = argparse.Namespace(config=config_path, prepared=prepared, run_dir=root / "run", template=raw / "submission.csv",
                                      expected_users=3, synthetic_fixture=True)
            run(args)
            summary = json.loads((args.run_dir / "summary.json").read_text())
            self.assertEqual(summary["status"], "complete_synthetic_fixture")
            self.assertEqual(summary["completed_trials"], 48)
            self.assertEqual(summary["failed_trials"], 0)
            self.assertEqual(before, sha256(base / "raw.sqlite"))
            for block in config["slices"]:
                self.assertTrue(json.loads((args.run_dir / block["name"] / "summary.json").read_text())["exact_AP_audits_passed"])
                for kind in ["primary", "secondary", "all_observed"]:
                    self.assertTrue(json.loads((args.run_dir / block["name"] / ("independent-AP-audit-" + kind + ".json")).read_text())["passed"])
            final = json.loads((args.run_dir / "final/submission-validation.json").read_text())
            self.assertEqual(final["rows"], 3)
            self.assertEqual(final["sha256"], sha256(args.run_dir / "final/submission.csv"))
            self.assertFalse(final["kaggle_uploaded"])
            self.assertLessEqual((args.run_dir / "selection.json").stat().st_mtime_ns,
                                 (args.run_dir / "final/model.pkl").stat().st_mtime_ns)
            self.assertFalse(json.loads((args.run_dir / "manifest.json").read_text())["source_changed_during_run"])
            with self.assertRaises(FileExistsError):
                run(args)
            unavailable_args = copy.copy(args)
            unavailable_args.run_dir = root / "als-unavailable"
            with patch("sweettv.v3_models.ALSModel", side_effect=ImportError("synthetic missing optional library")):
                run(unavailable_args)
            unavailable = json.loads((unavailable_args.run_dir / "summary.json").read_text())
            self.assertEqual(unavailable["completed_trials"], 39)
            self.assertEqual(unavailable["failed_trials"], 9)
            self.assertTrue((unavailable_args.run_dir / "weeks5/component-failures.json").exists())


if __name__ == "__main__":
    unittest.main()
