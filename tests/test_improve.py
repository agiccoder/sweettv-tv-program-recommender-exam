"""Synthetic tests of the bounded v2 proxy; never competition-quality results."""
import argparse
import copy
import datetime as dt
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from sweettv.data import build_interactions, import_data, sha256
from sweettv.improve import prepare_targets, read_history_scale, run
from test_pipeline_data import build_fixture, HUGE_USER


class PreferredTargetTests(unittest.TestCase):
    def test_unique_programmes_aggregate_before_top5_with_id_ties_and_common_cohort(self):
        frame = pd.DataFrame([
            {"user_id": HUGE_USER, "item_id": item, "seconds": seconds, "watches": watches}
            for item, seconds, watches in [(1, 10, 100), (2, 20, 1), (2, 30, 1), (3, 50, 3), (4, 40, 4), (5, 30, 5), (6, 20, 6), (7, 10, 7), (999, 9999, 9999)]
        ])
        cohort, targets, audit = prepare_targets(frame, list(range(1, 8)), [HUGE_USER, "absent"])
        self.assertEqual(cohort, [HUGE_USER])
        self.assertEqual(targets["primary"][HUGE_USER], {2, 3, 4, 5, 6})
        self.assertEqual(targets["secondary"][HUGE_USER], {1, 4, 5, 6, 7})
        self.assertEqual(targets["all_observed"][HUGE_USER], set(range(1, 8)))
        self.assertEqual(audit["details"][0]["primary_ordered_ids"], [2, 3, 4, 5, 6])
        self.assertEqual(audit["details"][0]["secondary_ordered_ids"], [1, 7, 6, 5, 4])
        self.assertEqual(audit["details"][0]["outside_pool_items"], 1)
        self.assertEqual(audit["excluded_no_eligible_targets"], 1)
        self.assertEqual(targets["primary"]["absent"], set())

    def test_fewer_than_five_labels_and_malformed_user_ids(self):
        frame = pd.DataFrame([{"user_id": HUGE_USER, "item_id": 1, "seconds": 120, "watches": 1}])
        _, targets, _ = prepare_targets(frame, [1, 2, 3, 4, 5], [HUGE_USER])
        self.assertEqual(targets["primary"][HUGE_USER], {1})
        malformed = frame.copy()
        malformed["user_id"] = 123
        with self.assertRaisesRegex(ValueError, "exact strings"):
            prepare_targets(malformed, [1, 2, 3, 4, 5], ["123"])


class ImprovementRunnerTests(unittest.TestCase):
    def test_complete_fourteen_method_freeze_selection_confirmation_and_new_final_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw, prepared = root/"raw", root/"prepared"
            build_fixture(raw)
            audit = import_data(raw, root/"raw.sqlite", root/"input-audit")
            build_interactions(root/"raw.sqlite", audit, prepared)
            source_config = Path(__file__).resolve().parents[1]/"config/improvement-v2.json"
            config = copy.deepcopy(json.loads(source_config.read_text()))
            block = {"train_start": "2020-01-01T00:00:00", "train_end": "2020-01-02T00:00:00", "target_end": "2020-01-03T00:00:00"}
            config["development"] = block
            config["confirmation"] = block
            config_path = root/"fixture-config.json"
            config_path.write_text(json.dumps(config))
            recency = root/"recency"
            recency.mkdir()
            for name, window, cutoff in [("development_train", "train", "2020-01-02"), ("confirmation_train", "train", "2020-01-02"), ("full", "full", "2020-01-03")]:
                feature = pd.read_parquet(prepared/f"{window}-interactions.parquet")
                feature["recent_seconds"] = feature.seconds*.5
                feature["max_event_end"] = dt.datetime.fromisoformat(cutoff).replace(tzinfo=dt.timezone.utc).timestamp()-1
                feature.to_parquet(recency/f"{name}.parquet", index=False)
            original_hashes = {str(path): sha256(path) for path in list(prepared.glob("*"))+list(raw.glob("*")) if path.is_file()}
            target = root/"v2-run"
            args = argparse.Namespace(config=str(config_path), run_dir=str(target), synthetic_fixture=True,
                                      template=str(raw/"submission.csv"), expected_users=3,
                                      development_prepared=str(prepared), confirmation_prepared=str(prepared), recency_dir=str(recency))
            run(args)
            summary = json.loads((target/"summary.json").read_text())
            self.assertEqual(summary["selected_method"], "original_popularity")
            self.assertEqual(summary["status"], "complete_synthetic_fixture")
            self.assertEqual(len(summary["development"]), 14)
            self.assertEqual(set(summary["confirmation"]), {"original_popularity", "original_content", "original_ppr_0.5"})
            self.assertTrue((target/"selection.json").is_file())
            self.assertLessEqual((target/"selection.json").stat().st_mtime_ns, (target/"confirmation/summary.json").stat().st_mtime_ns)
            self.assertFalse(json.loads((target/"selection.json").read_text())["confirmation_metrics_available_at_selection"])
            self.assertEqual(json.loads((target/"frozen-config.json").read_text()), config)
            for block_name in ["development", "confirmation"]:
                for kind in ["primary", "secondary", "all_observed"]:
                    self.assertTrue(json.loads((target/block_name/f"independent-ap-audit-{kind}.json").read_text())["passed"])
            self.assertEqual(original_hashes, {path: sha256(path) for path in original_hashes})
            self.assertTrue((target/"final/submission.csv").is_file())
            check = json.loads((target/"final/submission-validation.json").read_text())
            self.assertEqual(check["rows"], 3)
            self.assertEqual(check["sha256"], sha256(target/"final/submission.csv"))
            with self.assertRaises(FileExistsError):
                run(args)
            changed_args = copy.copy(args)
            changed_args.run_dir = str(root/"changed-source-run")
            with patch("sweettv.improve.source_hashes", side_effect=[{"source.py": "before"}, {"source.py": "after"}, {"source.py": "after"}]):
                with self.assertRaisesRegex(RuntimeError, "Source hashes changed"):
                    run(changed_args)
            changed_run = Path(changed_args.run_dir)
            self.assertEqual(json.loads((changed_run/"manifest.json").read_text())["status"], "failed")
            self.assertTrue(json.loads((changed_run/"manifest.json").read_text())["source_changed_during_run"])
            self.assertEqual(json.loads((changed_run/"failure.json").read_text())["exception_type"], "RuntimeError")
            self.assertFalse((changed_run/"summary.json").exists())

    def test_recency_scale_checks_source_seconds_and_future_events(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"features.parquet"
            train = pd.DataFrame([{"user_id": HUGE_USER, "item_id": 7, "seconds": 200, "watches": 2}])
            cutoff = "2020-01-02T00:00:00"
            cutoff_coordinate = dt.datetime.fromisoformat(cutoff).replace(tzinfo=dt.timezone.utc).timestamp()
            feature = train.copy()
            feature["recent_seconds"] = 50
            feature["max_event_end"] = cutoff_coordinate-1
            feature.to_parquet(path, index=False)
            self.assertEqual(read_history_scale(path, train, cutoff), {(HUGE_USER, 7): .25})
            feature["seconds"] = 199
            feature.to_parquet(path, index=False)
            with self.assertRaisesRegex(ValueError, "exact training"):
                read_history_scale(path, train, cutoff)
            feature["seconds"] = 200
            feature["max_event_end"] = cutoff_coordinate+1
            feature.to_parquet(path, index=False)
            with self.assertRaisesRegex(ValueError, "cutoff"):
                read_history_scale(path, train, cutoff)


if __name__ == "__main__":
    unittest.main()
