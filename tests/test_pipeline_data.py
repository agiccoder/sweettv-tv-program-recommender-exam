"""End-to-end synthetic input/import/join checks, never competition findings.

Fixture ``five_file_overlap_boundary_v1`` writes all five competition filenames
and executes the actual SQLite import and Parquet interaction export.  Dates,
users, programme IDs and measurements in this module are invented test data.
"""

import csv
import datetime as dt
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

import pandas as pd

from sweettv.data import FILES, build_interactions, import_data
from sweettv.metrics import evaluate
from sweettv.models import fit_popularity


FIXTURE_ID = "five_file_overlap_boundary_v1"
HUGE_USER = "000123456789012345678901234567890"
COLD_USER = "00000000000042"
SYNTHETIC_CALENDAR = {"train": "2020-01-01 00:00:00", "valid": "2020-01-02 00:00:00",
                      "future": "2020-01-03 00:00:00", "end": "2020-01-04 00:00:00"}
SCHEDULE_HEADER = [
    "channel_id", "channel_title", "start_time", "duration", "tv_show_title",
    "tv_show_id", "tv_show_category", "tv_show_genre_1", "tv_show_genre_2",
    "tv_show_genre_3", "year_of_production", "director", "actors",
]
VIEW_HEADER = ["user_id", "vsetv_id", "start_time", "stop_time", "duraton"]


def _iso(day, minute=0, second=0):
    return (dt.datetime(2020, 1, day) + dt.timedelta(minutes=minute, seconds=second)).isoformat(sep=" ")


def _schedule(day, channel, minute, duration_minutes, item):
    return [channel, f"Synthetic channel {channel}", _iso(day, minute), duration_minutes,
            f"Synthetic programme {item}", item, f"category-{item % 2}", "genre", "", "", "2020", "Director", "Actor"]


def _view(user, channel, start, end, *, declared_duration=None):
    seconds = (dt.datetime.fromisoformat(end) - dt.datetime.fromisoformat(start)).total_seconds()
    return [user, channel, start, end, seconds if declared_duration is None else declared_duration]


def build_fixture(directory, *, include_invalid_airing=False, nonfinite_view_duration=None,
                  include_calendar_spillovers=False, schedule_date_format="iso"):
    """Write complete, minute-unit synthetic data with known exact overlaps."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    schedules = {
        role: [_schedule(day, 1, i * 10, 10, day * 100 + i + 1) for i in range(6)]
        for role, day in [("train", 1), ("valid", 2), ("future", 3)]
    }
    schedules["train"] += [
        _schedule(1, 1, 0, 10, 101),  # Exact schedule duplicate.
        [1, "Synthetic channel 1", _iso(1, 0, 30), 8.5,
         "Synthetic programme 101", 101, "category-1", "genre", "", "", "2020", "Director", "Actor"],
        _schedule(1, 2, 23 * 60 + 50, 20, 150),  # Continues over validation boundary.
        _schedule(1, 3, 0, 10, 0),
        _schedule(1, 3, 10, 10, 160),
        _schedule(1, 4, 0, 10, 170),
        _schedule(1, 4, 2, 2, 171),  # Two minutes of conflicting nonzero labels.
        _schedule(1, 4, 10, 10, 172),
    ]
    schedules["valid"] += [
        _schedule(2, 2, 0, 10, 150),
        _schedule(2, 2, 10, 10, 250),
        _schedule(2, 5, 23 * 60 + 50, 20, 260),  # Continues into future; views are clipped.
    ]
    schedules["future"] += [_schedule(3, 5, 0, 10, 360), _schedule(3, 5, 10, 10, 361)]
    if include_invalid_airing:
        schedules["train"] += [_schedule(1, 6, 3, 0, 999), _schedule(1, 6, 4, -1, 998)]
    views = [
        _view(HUGE_USER, 1, _iso(1, 0), _iso(1, 1)),
        _view(HUGE_USER, 1, _iso(1, 0), _iso(1, 1)),  # Exact view duplicate.
        _view(HUGE_USER, 1, _iso(1, 0, 30), _iso(1, 1, 30)),
        _view(HUGE_USER, 1, _iso(1, 2), _iso(1, 3)),  # 90+60=150 union seconds.
        _view(HUGE_USER, 1, _iso(2, 10), _iso(2, 12)),
        _view(COLD_USER, 1, _iso(2, 0), _iso(2, 3)),
        _view("boundary", 2, _iso(1, 23 * 60 + 59), _iso(2, 1)),
        _view("zero-only", 3, _iso(1, 0), _iso(1, 2)),
        _view("conflict", 4, _iso(1, 1), _iso(1, 5)),
        _view("wrong-channel", 999, _iso(1, 0), _iso(1, 2)),
        _view("exact-end", 1, _iso(1, 60), _iso(1, 62)),
        _view("future-crossing", 5, _iso(2, 23 * 60 + 58), _iso(3, 2)),
        _view("future-only", 1, _iso(3, 0), _iso(3, 10)),
    ]
    if nonfinite_view_duration is not None:
        views[0][-1] = nonfinite_view_duration
    if include_calendar_spillovers:
        # Unique programme metadata appears only in the wrong nominal source
        # file's tail; calendar filtering, rather than file membership, decides.
        schedules["train"].append(_schedule(2, 6, 30, 10, 190))
        schedules["valid"].append(_schedule(3, 6, 30, 10, 290))
        views += [
            _view(COLD_USER, 6, _iso(2, 30), _iso(2, 33)),
            _view("tail-future-user", 6, _iso(3, 30), _iso(3, 35)),
        ]
    for role, data in schedules.items():
        if schedule_date_format == "dd.mm.yyyy":
            for row in data:
                row[2] = dt.datetime.fromisoformat(row[2]).strftime("%d.%m.%Y %H:%M:%S")
        elif schedule_date_format != "iso":
            raise ValueError("Unsupported synthetic schedule_date_format")
        with (directory / FILES[role]).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(SCHEDULE_HEADER)
            writer.writerows(data)
    with (directory / FILES["views"]).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(VIEW_HEADER)
        writer.writerows(views)
    with (directory / FILES["template"]).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["user_id", "tv_show_id"])
        writer.writerows([[user, "0 0 0 0 0"] for user in [HUGE_USER, COLD_USER, "future-only"]])


class PipelineDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.raw = self.directory / "raw"
        self.db = self.directory / "processed" / "test.sqlite"
        self.evidence = self.directory / "evidence"
        self.interactions = self.directory / "processed" / "interactions"

    def run_pipeline(self, *, calendar_boundaries=None, **fixture_options):
        build_fixture(self.raw, **fixture_options)
        audit = import_data(self.raw, self.db, self.evidence)
        protocol, join_audit = build_interactions(self.db, audit, self.interactions,
                                                calendar_boundaries=calendar_boundaries)
        frames = {window: pd.read_parquet(self.interactions / f"{window}-interactions.parquet")
                  for window in ["train", "valid", "full"]}
        return audit, protocol, join_audit, frames

    def test_import_infers_minutes_preserves_exact_ids_and_audits_duplicates(self):
        audit, _, _, frames = self.run_pipeline()
        for role in ["train", "valid", "future"]:
            self.assertEqual(audit["files"][role]["duration_scale_seconds"], 60)
            self.assertIsNone(audit["files"][role]["duration_override"])
            self.assertGreater(audit["files"][role]["duration_unit_test"]["minutes"]["endpoint_match_fraction"], 0.25)
        self.assertEqual(audit["view_duplicate_rows"], 1)
        self.assertEqual(audit["files"]["train"]["duplicate_airing_rows"], 1)
        self.assertEqual(audit["files"]["train"]["unlabelled_rows"], 1)
        self.assertIn(HUGE_USER, set(frames["train"]["user_id"]))
        with sqlite3.connect(self.db) as connection:
            stored = connection.execute("SELECT DISTINCT user_id FROM views WHERE user_id=?", (HUGE_USER,)).fetchone()
        self.assertEqual(stored, (HUGE_USER,))

    def test_actual_export_joins_fragments_without_duplicate_seconds(self):
        _, _, audit, frames = self.run_pipeline()
        train = frames["train"].set_index(["user_id", "item_id"])
        self.assertEqual(set(train.index), {(HUGE_USER, 101), ("conflict", 170)})
        self.assertEqual(train.loc[(HUGE_USER, 101), "seconds"], 150)
        self.assertEqual(train.loc[(HUGE_USER, 101), "watches"], 1)
        self.assertEqual(train.loc[("conflict", 170), "seconds"], 120)
        self.assertEqual(audit["train"]["ambiguous_labels_seconds"], 120)
        self.assertEqual(audit["train"]["unlabelled_seconds"], 120)
        self.assertEqual(audit["train"]["outside_schedule_seconds"], 240)
        self.assertEqual(audit["train"]["users_without_usable_interactions"], 4)

    def test_boundary_clipping_and_future_views_are_excluded_from_preferences(self):
        _, protocol, _, frames = self.run_pipeline()
        self.assertFalse(protocol["future_views_used"])
        for window in ["train", "valid", "full"]:
            self.assertNotIn("future-only", set(frames[window]["user_id"]))
            self.assertFalse(set(frames[window]["item_id"]) & {301, 302, 303, 304, 305, 306, 360, 361})
        self.assertNotIn("boundary", set(frames["train"]["user_id"]))
        self.assertNotIn("boundary", set(frames["valid"]["user_id"]))
        full = frames["full"].set_index(["user_id", "item_id"])
        valid = frames["valid"].set_index(["user_id", "item_id"])
        self.assertEqual(full.loc[("boundary", 150), "seconds"], 120)
        self.assertEqual(full.loc[("future-crossing", 260), "seconds"], 120)
        self.assertEqual(valid.loc[("future-crossing", 260), "seconds"], 120)
        observed = json.loads((self.interactions / "observed-users.json").read_text())
        self.assertNotIn("future-only", observed)

    def test_cold_validation_user_survives_actual_exports_and_fixed_cohort_evaluation(self):
        _, _, _, frames = self.run_pipeline()
        self.assertNotIn(COLD_USER, set(frames["train"]["user_id"]))
        self.assertIn(COLD_USER, set(frames["valid"]["user_id"]))
        candidates = json.loads((self.interactions / "candidates.json").read_text())["valid"]
        relevant = {
            user: set(group["item_id"]) & set(candidates)
            for user, group in frames["valid"].groupby("user_id", sort=True)
        }
        cohort = sorted(user for user in relevant if relevant[user])
        model = fit_popularity(frames["train"])
        predictions = {user: [row["item_id"] for row in model.recommend(user, candidates)] for user in cohort}
        result = evaluate(predictions, relevant, cohort)
        self.assertIn(COLD_USER, result["per_user_ap"])
        self.assertEqual(result["cohort_size"], 3)
        self.assertEqual(len(predictions[COLD_USER]), 5)
        self.assertTrue(set(predictions[COLD_USER]).issubset(candidates))

    def test_nonpositive_airings_are_not_candidates_or_metadata(self):
        _, _, _, _ = self.run_pipeline(include_invalid_airing=True)
        candidates = json.loads((self.interactions / "candidates.json").read_text())
        metadata = json.loads((self.interactions / "metadata.json").read_text())
        self.assertFalse({998, 999} & set(candidates["train"]))
        self.assertFalse({"998", "999"} & set(metadata["train"]))

    def test_nonfinite_declared_view_duration_is_rejected_during_import(self):
        for bad_value in ["nan", "inf", "-inf"]:
            with self.subTest(declared_duration=bad_value):
                raw = self.directory / f"raw-{bad_value}"
                build_fixture(raw, nonfinite_view_duration=bad_value)
                with self.assertRaisesRegex(ValueError, "finite"):
                    import_data(raw, self.directory / f"{bad_value}.sqlite", self.evidence)

    def test_nonfinite_view_endpoints_are_already_rejected_during_import(self):
        for column, index in [("start_time", 2), ("stop_time", 3)]:
            for bad_value in ["nan", "inf", "-inf"]:
                with self.subTest(column=column, endpoint=bad_value):
                    fixture_name = f"{column}-{bad_value}"
                    raw = self.directory / fixture_name
                    build_fixture(raw)
                    path = raw / FILES["views"]
                    with path.open(newline="", encoding="utf-8") as stream:
                        records = list(csv.reader(stream))
                    records[1][index] = bad_value
                    with path.open("w", newline="", encoding="utf-8") as stream:
                        csv.writer(stream).writerows(records)
                    with self.assertRaisesRegex(ValueError, "timestamp"):
                        import_data(raw, self.directory / f"{fixture_name}.sqlite", self.evidence)

    def test_actual_day_first_schedule_format_and_calendar_spillovers_do_not_leak(self):
        audit, protocol, _, frames = self.run_pipeline(
            calendar_boundaries=SYNTHETIC_CALENDAR,
            include_calendar_spillovers=True, schedule_date_format="dd.mm.yyyy",
        )
        expected_valid_start = dt.datetime(2020, 1, 2, tzinfo=dt.timezone.utc).timestamp()
        self.assertEqual(audit["files"]["valid"]["start_min"], expected_valid_start)
        self.assertEqual(audit["files"]["train"]["first_record"]["start_time"], "01.01.2020 00:00:00")
        self.assertEqual(protocol["explicit_calendar_boundaries"], SYNTHETIC_CALENDAR)
        candidates = json.loads((self.interactions / "candidates.json").read_text())
        metadata = json.loads((self.interactions / "metadata.json").read_text())
        self.assertNotIn(190, candidates["train"])
        self.assertNotIn("190", metadata["train"])
        self.assertIn(190, candidates["valid"])
        self.assertIn("190", metadata["valid"])
        self.assertNotIn(290, candidates["valid"])
        self.assertNotIn("290", metadata["valid"])
        self.assertNotIn("tail-future-user", set(frames["full"]["user_id"]))
        self.assertNotIn(190, set(frames["train"]["item_id"]))
        self.assertEqual(set(frames["valid"].query("item_id == 190")["user_id"]), {COLD_USER})
        self.assertFalse(any(290 in set(frame["item_id"]) for frame in frames.values()))

    def test_calendar_spillovers_require_explicit_boundaries_and_us_slash_dates_are_rejected(self):
        build_fixture(self.raw, include_calendar_spillovers=True, schedule_date_format="dd.mm.yyyy")
        audit = import_data(self.raw, self.db, self.evidence)
        with self.assertRaisesRegex(ValueError, "overlap in start dates"):
            build_interactions(self.db, audit, self.interactions)
        invalid_raw = self.directory / "unsupported-date-format"
        build_fixture(invalid_raw)
        path = invalid_raw / FILES["train"]
        with path.open(newline="", encoding="utf-8") as stream:
            records = list(csv.reader(stream))
        records[1][2] = "01/02/2020 00:00:00"
        with path.open("w", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerows(records)
        with self.assertRaises(ValueError):
            import_data(invalid_raw, self.directory / "unsupported.sqlite", self.evidence)


if __name__ == "__main__":
    unittest.main()
