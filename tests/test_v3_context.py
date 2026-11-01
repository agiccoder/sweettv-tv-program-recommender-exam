"""Hand-checkable raw context/cutoff/schedule fixtures; no quality claims."""
import datetime as dt
import importlib.util
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

import numpy as np
import pandas as pd
from scipy import sparse

from sweettv.data import build_interactions, import_data, sha256, timestamp
from sweettv.improved_models import fit_improved_ppr
from sweettv.v3_context import (ContextState, availability_matrix, calendar_parts,
                               collect_contexts, interval_profile)
from test_pipeline_data import build_fixture, HUGE_USER

ROOT = Path(__file__).resolve().parents[1]
MONDAY = timestamp("2020-03-09T00:00:00")


class ContextTests(unittest.TestCase):
    def test_calendar_union_midnight_and_cutoff_by_hand(self):
        intervals = [(MONDAY + 23 * 3600, MONDAY + 26 * 3600)] * 2
        expected = {(0, 5): 3600., (1, 0): 3600.}
        self.assertEqual(interval_profile(intervals, MONDAY, MONDAY + 25 * 3600), expected)
        self.assertEqual(interval_profile(list(reversed(intervals)), MONDAY, MONDAY + 25 * 3600), expected)
        self.assertEqual(list(calendar_parts(MONDAY + 3 * 3600, MONDAY + 5 * 3600)), [(0, 0, 3600.), (0, 1, 3600.)])

    def test_availability_duplicates_conflicts_duration_mean_and_unknown_channels(self):
        airings = {1: [(MONDAY, MONDAY + 3600, 1), (MONDAY + 3600, MONDAY + 7200, 2),
                       (MONDAY + 3600, MONDAY + 5400, 3), (MONDAY, MONDAY + 7200, 0)],
                   9: [(MONDAY, MONDAY + 3600, 4)]}
        a, items, totals, audit = availability_matrix(airings, [1, 2, 3, 4, 5], [1], MONDAY, MONDAY + 86400)
        self.assertEqual(items, [1, 2, 3, 4, 5])
        np.testing.assert_array_equal(totals, [3600, 1800, 0, 3600, 0])
        self.assertEqual(a[0, 0], 1)
        self.assertEqual(a[1, 0], 1)
        self.assertEqual(a[3].nnz, 0)
        duplicated = {channel: list(reversed(rows + rows)) for channel, rows in airings.items()}
        b, _, b_totals, _ = availability_matrix(duplicated, [1, 2, 3, 4, 5], [1], MONDAY, MONDAY + 86400)
        self.assertEqual((a != b).nnz, 0)
        np.testing.assert_array_equal(totals, b_totals)
        self.assertEqual(audit["ambiguous_labels_seconds"], 1800)

    def test_smoothing_uses_training_global_only_and_no_history_has_no_native_score(self):
        values = sparse.csr_matrix(([100., 300.], ([0, 1], [0, 1])), shape=(2, 42))
        state = ContextState(values, ["0001", "u2"], [1], MONDAY, MONDAY + 86400, smoothing_seconds=100)
        exposure = sparse.csr_matrix(([1., 1.], ([0, 1], [0, 1])), shape=(2, 42))
        np.testing.assert_allclose(state.scores("0001", exposure), [.625, .375])
        self.assertIsNone(state.scores("absent", exposure))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "context"
            state.save(path)
            restored = ContextState.load(path)
            np.testing.assert_array_equal(restored.scores("0001", exposure), state.scores("0001", exposure))

    def test_actual_prepare_later_views_cannot_change_early_profile_or_graph(self):
        spec = importlib.util.spec_from_file_location("prepare_v3_fixture", ROOT / "scripts/prepare_improvement_v3.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            base = directory / "base"
            base.mkdir()
            raw = directory / "raw"
            build_fixture(raw)
            audit = import_data(raw, base / "raw.sqlite", base / "audit")
            build_interactions(base / "raw.sqlite", audit, base / "prepared")
            config = json.loads((ROOT / "config/improvement-v3.json").read_text())
            for block in config["slices"]:
                block.update({"train_start": "2020-01-01T00:00:00", "train_end": "2020-01-02T00:00:00", "target_end": "2020-01-03T00:00:00"})
            config["final"] = {"train_start": "2020-01-01T00:00:00", "train_end": "2020-01-03T00:00:00", "target_end": "2020-01-04T00:00:00"}
            config_path = directory / "fixture-config.json"
            config_path.write_text(json.dumps(config))
            one = directory / "one"
            before = sha256(base / "raw.sqlite")
            module.prepare(base, one, config_path, raw / "submission.csv", expected_users=3)
            self.assertEqual(before, sha256(base / "raw.sqlite"))
            early = ContextState.load(one / "context/weeks5")
            self.assertIn("zero-only", early.users)  # ID=0 airtime remains raw-channel context.
            self.assertNotIn("future-only", early.users)
            self.assertNotIn("zero-only", set(pd.read_parquet(one / "weeks5/train-interactions.parquet").user_id))
            with sqlite3.connect(base / "raw.sqlite") as conn:
                start = timestamp("2020-01-03T01:00:00")
                conn.execute("INSERT INTO views VALUES(?,?,?,?,?)", ("future-extra", 987654, start, start + 1000, 1000))
            two = directory / "two"
            module.prepare(base, two, config_path, raw / "submission.csv", expected_users=3)
            later = ContextState.load(two / "context/weeks5")
            self.assertEqual(early.users, later.users)
            self.assertEqual(early.channels, later.channels)
            self.assertEqual((early.counts != later.counts).nnz, 0)
            a = fit_improved_ppr(pd.read_parquet(one / "weeks5/train-interactions.parquet"), alpha=.25)
            b = fit_improved_ppr(pd.read_parquet(two / "weeks5/train-interactions.parquet"), alpha=.25)
            self.assertEqual((a.transition != b.transition).nnz, 0)
            self.assertEqual((a.restart_rows != b.restart_rows).nnz, 0)


if __name__ == "__main__":
    unittest.main()
