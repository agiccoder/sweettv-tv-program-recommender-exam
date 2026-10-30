"""Actual CLI integration on invented fixtures; these are not exam results."""

import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pandas as pd

from sweettv.data import FILES
from sweettv.metrics import audit_exported_ap
from sweettv.outputs import validate_submission
from test_pipeline_data import build_fixture, COLD_USER, HUGE_USER, SYNTHETIC_CALENDAR


class CLIIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.source_root = Path(__file__).resolve().parents[1]
        self.environment = dict(os.environ)
        self.environment["PYTHONPATH"] = str(self.source_root / "src")

    def command(self, raw, run):
        return [sys.executable, "-m", "sweettv.cli", "run", "--raw-dir", str(raw),
                "--run-dir", str(run), "--expected-users", "3", "--synthetic-fixture"]

    def run_pipeline(self, label):
        raw = self.directory / f"raw-{label}"
        run = self.directory / f"run-{label}"
        build_fixture(raw)
        result = subprocess.run(self.command(raw, run), cwd=self.source_root,
                                env=self.environment, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
        return raw, run

    @staticmethod
    def hashes(directory):
        return {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in directory.rglob("*") if path.is_file()}

    def test_complete_cli_artifacts_submission_and_independent_ap_audit(self):
        raw = self.directory / "raw"
        run = self.directory / "run"
        build_fixture(raw)
        before = self.hashes(raw)
        result = subprocess.run(self.command(raw, run), cwd=self.source_root,
                                env=self.environment, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
        self.assertEqual(before, self.hashes(raw))
        manifest = json.loads((run / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "complete_synthetic_fixture")
        self.assertTrue(manifest["synthetic_fixture"])
        self.assertFalse(manifest["external_uploads"])
        self.assertIn("src/sweettv/data.py", manifest["source_hashes"])
        self.assertFalse((run / "failure.json").exists())
        summary = json.loads((run / "validation-summary.json").read_text())
        self.assertEqual(set(summary["methods"]), {"popularity", "content", "ppr_0.5", "ppr_0.85"})
        self.assertEqual(summary["cohort_size"], 2)
        cohort = json.loads((run / "cohort.json").read_text())
        self.assertEqual(cohort["eligible_zero_history_users"], 1)
        self.assertEqual(cohort["user_ids"], [HUGE_USER, COLD_USER])
        self.assertEqual(cohort["excluded_no_eligible_targets"], 1)
        ap_audit = audit_exported_ap(run / "per-user-ap.csv", run / "validation-summary.json")
        self.assertTrue(ap_audit["passed"])
        self.assertEqual(set(ap_audit["methods"]), set(summary["methods"]))
        for values in ap_audit["methods"].values():
            self.assertEqual(values["cohort_size"], 2)
        candidates = json.loads((run / "prepared" / "candidates.json").read_text())["future"]
        checked = validate_submission(raw / FILES["template"], run / "final" / "submission.csv", set(candidates), expected_users=3)
        saved_check = json.loads((run / "final" / "submission-validation.json").read_text())
        self.assertEqual(checked["sha256"], saved_check["sha256"])
        self.assertEqual(checked["rows"], 3)
        self.assertEqual(saved_check["selected_method"], summary["selected_method"])
        with (run / "final" / "recommendations.csv").open(newline="") as stream:
            recs = list(csv.DictReader(stream))
        self.assertEqual(len(recs), 15)
        self.assertTrue(all(int(row["item_id"]) in candidates for row in recs))
        self.assertTrue(all(row["title"].startswith("Synthetic programme ") for row in recs))
        self.assertTrue(all(row["category"].startswith("category-") for row in recs))
        for name in summary["methods"]:
            self.assertTrue((run / "models" / f"{name}.pkl").is_file())
            self.assertTrue((run / "diagnostics" / f"{name}.json").is_file())
        self.assertTrue((run / "final" / "model.pkl").is_file())

    def test_future_view_changes_do_not_change_preferences_ap_or_submission(self):
        raw, first = self.run_pipeline("future-invariance")
        with (raw / FILES["views"]).open("a", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerows([
                [HUGE_USER, 1, "2020-01-03 00:00:00", "2020-01-03 02:00:00", 7200],
                [COLD_USER, 5, "2020-01-03 00:00:00", "2020-01-03 01:00:00", 3600],
                ["new-future-user", 1, "2020-01-03 00:00:00", "2020-01-03 00:10:00", 600],
            ])
        before = self.hashes(raw)
        second = self.directory / "run-future-mutated"
        result = subprocess.run(self.command(raw, second), cwd=self.source_root,
                                env=self.environment, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
        self.assertEqual(before, self.hashes(raw))
        for window in ["train", "valid", "full"]:
            pd.testing.assert_frame_equal(
                pd.read_parquet(first / "prepared" / f"{window}-interactions.parquet"),
                pd.read_parquet(second / "prepared" / f"{window}-interactions.parquet"),
            )
        self.assertEqual((first / "per-user-ap.csv").read_bytes(), (second / "per-user-ap.csv").read_bytes())
        self.assertEqual((first / "final" / "submission.csv").read_bytes(), (second / "final" / "submission.csv").read_bytes())
        self.assertEqual(audit_exported_ap(first / "per-user-ap.csv"), audit_exported_ap(second / "per-user-ap.csv"))
        first_audit = json.loads((first / "audit" / "data-audit.json").read_text())
        second_audit = json.loads((second / "audit" / "data-audit.json").read_text())
        self.assertNotEqual(first_audit["files"]["views"]["sha256"], second_audit["files"]["views"]["sha256"])

    def test_cli_refuses_to_overwrite_an_existing_run(self):
        raw, run = self.run_pipeline("immutable")
        original_manifest = (run / "manifest.json").read_bytes()
        result = subprocess.run(self.command(raw, run), cwd=self.source_root,
                                env=self.environment, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Immutable run directory already exists", result.stderr)
        self.assertEqual((run / "manifest.json").read_bytes(), original_manifest)

    def test_cli_calendar_config_handles_real_format_and_source_file_spillovers(self):
        raw = self.directory / "raw-calendar"
        target = self.directory / "run-calendar"
        build_fixture(raw, include_calendar_spillovers=True, schedule_date_format="dd.mm.yyyy")
        calendar = self.directory / "calendar.json"
        calendar.write_text(json.dumps(SYNTHETIC_CALENDAR))
        before = self.hashes(raw)
        result = subprocess.run(self.command(raw, target) + ["--calendar-config", str(calendar)],
                                cwd=self.source_root, env=self.environment, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
        self.assertEqual(before, self.hashes(raw))
        candidates = json.loads((target / "prepared" / "candidates.json").read_text())
        metadata = json.loads((target / "prepared" / "metadata.json").read_text())
        self.assertNotIn(190, candidates["train"])
        self.assertNotIn("190", metadata["train"])
        self.assertIn(190, candidates["valid"])
        self.assertIn("190", metadata["valid"])
        self.assertNotIn(290, candidates["valid"])
        self.assertNotIn("290", metadata["valid"])
        self.assertTrue(audit_exported_ap(target / "per-user-ap.csv", target / "validation-summary.json")["passed"])
        manifest = json.loads((target / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "complete_synthetic_fixture")


if __name__ == "__main__":
    unittest.main()
