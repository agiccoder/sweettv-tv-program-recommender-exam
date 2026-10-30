#!/usr/bin/env python3
"""Reproduce v2 preparation from an existing audited original pipeline run."""
import argparse
import json
from pathlib import Path
import sqlite3
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sweettv.data import build_interactions, dump_json, sha256
from sweettv.time_features import build_recency_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config/improvement-v2.json")
    args = parser.parse_args()
    base, output = args.base_run.resolve(), args.output_root.resolve()
    if output.exists():
        raise FileExistsError("Preparation is immutable; choose a new output directory")
    config = json.loads(args.config.read_text())
    audit_path, database = base / "audit/data-audit.json", base / "raw.sqlite"
    audit = json.loads(audit_path.read_text())
    before = sha256(database)
    original_connect = sqlite3.connect

    def readonly_connect(path, *unused_args, **unused_kwargs):
        if Path(path).resolve() != database:
            raise ValueError("Unexpected database in original preparation helper")
        connection = original_connect(database.as_uri() + "?mode=ro", uri=True)
        connection.execute("PRAGMA query_only=ON")
        return connection

    development = config["development"]
    calendar = {"train": development["train_start"], "valid": development["train_end"],
                "future": development["target_end"], "end": "2020-10-19T00:00:00"}
    with patch("sweettv.data.sqlite3.connect", side_effect=readonly_connect):
        build_interactions(database, audit, output / "development", threshold_seconds=120,
                           calendar_boundaries=calendar)
    windows = {"development_train": (development["train_start"], development["train_end"]),
               "confirmation_train": (config["confirmation"]["train_start"], config["confirmation"]["train_end"]),
               "full": (config["confirmation"]["train_start"], config["confirmation"]["target_end"])}
    build_recency_features(database, output / "recency", audit, windows=windows,
                           half_life_days=config["recency_variant"]["half_life_days"],
                           references={"development_train": output / "development/train-interactions.parquet",
                                       "confirmation_train": base / "prepared/train-interactions.parquet",
                                       "full": base / "prepared/full-interactions.parquet"})
    after = sha256(database)
    if before != after:
        raise AssertionError("The read-only source database changed")
    dump_json(output / "preparation-manifest.json", {
        "status": "complete", "script_sha256": sha256(__file__), "config_sha256": sha256(args.config),
        "base_audit_sha256": sha256(audit_path), "database_sha256": before, "database_unchanged": True,
        "development_calendar": calendar, "fitted_models": False,
        "unused_development_outputs": ["full-interactions.parquet", "future candidate/metadata sections"],
        "restriction": "Early tuning uses train and valid sections only. Future schedule outputs from the original helper are unused. No future viewing is used."})
    print(json.dumps({"status": "complete", "output_root": str(output), "database_unchanged": True}))


if __name__ == "__main__":
    main()
