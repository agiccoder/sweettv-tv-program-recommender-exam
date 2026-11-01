#!/usr/bin/env python3
"""Immutable v3 preparation from the read-only audited raw database."""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import time
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sweettv.data import build_interactions, dump_json, sha256, timestamp
from sweettv.improve import prepare_targets, read_template
from sweettv.time_features import build_recency_features
from sweettv.v3_context import ContextState, collect_contexts, schedule_features
from scipy import sparse
import pandas as pd


def prepare(base, output, config_path, template, expected_users=1260):
    base, output, config_path = Path(base).resolve(), Path(output).resolve(), Path(config_path).resolve()
    if output.exists():
        raise FileExistsError("V3 preparation directory must be new")
    output.mkdir(parents=True)
    started = time.perf_counter()
    config = json.loads(config_path.read_text())
    database = base / "raw.sqlite"
    audit_path = base / "audit/data-audit.json"
    audit = json.loads(audit_path.read_text())
    relevant = [ROOT / "src/sweettv/data.py", ROOT / "src/sweettv/time_features.py",
                ROOT / "src/sweettv/v3_context.py", Path(__file__), config_path]
    hashes = {str(p): sha256(p) for p in relevant}
    manifest = {"status": "running", "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "config_sha256": sha256(config_path), "database_sha256_before": sha256(database),
                "source_hashes": hashes, "base_audit_sha256": sha256(audit_path), "quality_metrics_computed": False,
                "sqlite_access": "mode=ro and query_only=ON", "external_uploads": False}
    dump_json(output / "preparation-manifest.json", manifest)
    original_connect = sqlite3.connect

    def ro_connect(path, *args, **kwargs):
        if Path(path).resolve() != database:
            raise ValueError("Unexpected source database")
        conn = original_connect(database.as_uri() + "?mode=ro", uri=True)
        conn.execute("PRAGMA query_only=ON")
        return conn

    try:
        for block in config["slices"]:
            destination = output / block["name"]
            print(json.dumps({"phase": "original_join", "slice": block["name"]}), flush=True)
            calendar = {"train": block["train_start"], "valid": block["train_end"],
                        "future": block["target_end"], "end": config["final"]["target_end"]}
            with patch("sweettv.data.sqlite3.connect", side_effect=ro_connect):
                build_interactions(database, audit, destination, threshold_seconds=config["threshold_seconds"],
                                   calendar_boundaries=calendar)
        windows = {block["name"]: (block["train_start"], block["train_end"]) for block in config["slices"]}
        windows["full"] = (config["final"]["train_start"], config["final"]["train_end"])
        print(json.dumps({"phase": "recency_all_cutoffs"}), flush=True)
        references = {block["name"]: output / block["name"] / "train-interactions.parquet" for block in config["slices"]}
        references["full"] = base / "prepared/full-interactions.parquet"
        build_recency_features(database, output / "recency", audit, windows=windows,
                              half_life_days=config["recency"]["half_life_days"], references=references)
        print(json.dumps({"phase": "raw_channel_profiles"}), flush=True)
        context_audit = collect_contexts(database, output / "context", windows, config["context"], audit)
        old_full = json.loads((base / "prepared/join-audit.json").read_text())["full"]
        for key in ["view_seconds_after_union", "unlabelled_seconds", "labelled_seconds", "ambiguous_labels_seconds", "outside_schedule_seconds"]:
            if context_audit["full_raw_time_audit"].get(key, 0.) != old_full.get(key, 0.):
                raise AssertionError("Independent full raw-time audit differs: " + key)
        users = read_template(template, expected_users)
        for block in config["slices"]:
            directory = output / block["name"]
            candidates = json.loads((directory / "candidates.json").read_text())["valid"]
            state = ContextState.load(output / "context" / block["name"])
            matrix, items, totals, availability_audit = schedule_features(
                database, audit, state, candidates, timestamp(block["train_end"]), timestamp(block["target_end"]))
            sparse.save_npz(directory / "availability.npz", matrix)
            dump_json(directory / "availability.json", {"items": items, "airtime_seconds": totals.tolist(),
                      "audit": availability_audit, "scoring_fields": ["channel", "start", "duration", "item_id"]})
            cohort, _, targets = prepare_targets(pd.read_parquet(directory / "valid-interactions.parquet"), candidates, users)
            targets["frozen_before_model_scoring"] = True
            dump_json(directory / "cohort-and-targets.json", targets)
            print(json.dumps({"phase": "slice_prepared", "slice": block["name"], "cohort": len(cohort), "candidates": len(items)}), flush=True)
        final = output / "final"
        final.mkdir()
        shutil.copy2(base / "prepared/full-interactions.parquet", final / "train-interactions.parquet")
        records = json.loads((base / "prepared/metadata.json").read_text())
        train_metadata = {item: dict(record) for item, record in records["train"].items()}
        for item, record in records["valid"].items():
            previous = train_metadata.setdefault(item, {})
            for key, value in record.items():
                if not previous.get(key) and value:
                    previous[key] = value
        dump_json(final / "metadata.json", {"train": train_metadata})
        candidates = json.loads((base / "prepared/candidates.json").read_text())["future"]
        dump_json(final / "candidates.json", {"valid": candidates})
        state = ContextState.load(output / "context/full")
        matrix, items, totals, availability_audit = schedule_features(database, audit, state, candidates,
            timestamp(config["final"]["train_end"]), timestamp(config["final"]["target_end"]), future=True)
        sparse.save_npz(final / "availability.npz", matrix)
        dump_json(final / "availability.json", {"items": items, "airtime_seconds": totals.tolist(), "audit": availability_audit})
        dump_json(final / "protocol.json", {"train": [timestamp(config["final"]["train_start"]), timestamp(config["final"]["train_end"])],
                  "target": [timestamp(config["final"]["train_end"]), timestamp(config["final"]["target_end"])],
                  "future_content_features_used": False, "future_view_preferences_used": False})
        if hashes != {str(p): sha256(p) for p in relevant}:
            raise RuntimeError("Preparation source/config changed during run")
        manifest["database_sha256_after"] = sha256(database)
        if manifest["database_sha256_before"] != manifest["database_sha256_after"]:
            raise RuntimeError("Source raw.sqlite changed")
        manifest.update({"status": "complete", "original_database_unchanged": True,
                         "unused_helper_outputs": ["per-slice full-interactions.parquet", "per-slice future candidate/metadata sections"],
                         "model_input_hashes": {str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob("*"))
                             if p.is_file() and p.name != "preparation-manifest.json"}})
    except Exception as error:
        manifest["status"] = "failed"
        dump_json(output / "failure.json", {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
        raise
    finally:
        manifest["wall_seconds"] = time.perf_counter() - started
        manifest["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        dump_json(output / "preparation-manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run", type=Path, default=ROOT / "experiments/competition-20261030-03")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config/improvement-v3.json")
    parser.add_argument("--template", type=Path, default=ROOT / "data/raw/extracted/submission.csv")
    parser.add_argument("--expected-users", type=int, default=1260)
    args = parser.parse_args()
    result = prepare(args.base_run, args.output_root, args.config, args.template, args.expected_users)
    print(json.dumps({"status": result["status"], "seconds": result["wall_seconds"], "database_unchanged": result["original_database_unchanged"]}))


if __name__ == "__main__":
    main()
