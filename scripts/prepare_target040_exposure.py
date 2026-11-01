#!/usr/bin/env python3
"""Cutoff-specific broadcast exposure from permitted schedule fields only.

No view log is queried. Historical and target schedule opportunities are
separate; target availability is never interpreted as future viewing.
"""
import argparse
from collections import defaultdict
import datetime as dt
import hashlib
import json
from pathlib import Path
import sqlite3
import time

from sweettv.data import channel_segments, sha256, timestamp

ROOT = Path(__file__).resolve().parents[1]


def exposure(airings, lower, upper, threshold=120.):
    clipped = [(max(start, lower), min(end, upper), int(item)) for start, end, item in airings
               if start < upper and end > lower and end > start]
    seconds, per_airing = defaultdict(float), defaultdict(float)
    for start, end, item, original_start, original_end, reason in channel_segments(clipped):
        if item > 0 and reason == "labelled":
            seconds[item] += end-start
            per_airing[item, original_start, original_end] += end-start
    counts = defaultdict(int)
    for (item, _, _), value in per_airing.items():
        if value >= threshold:
            counts[item] += 1
    return dict(seconds), dict(counts)


def prepare(output, config_path=None, prepared=None):
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    start = time.perf_counter()
    base = ROOT / "experiments/competition-20261030-03"
    database = base / "raw.sqlite"
    audit = json.loads((base / "audit/data-audit.json").read_text())
    config_path = Path(config_path or ROOT / "config/improvement-v3.json")
    prepared = Path(prepared or ROOT / "experiments/improvement-v3-data-01")
    config = json.loads(config_path.read_text())
    before = sha256(database)
    bounds = {}
    for block in config["slices"] + [{"name": "final", **config["final"]}]:
        for kind, lower, upper in [("train", block["train_start"], block["train_end"]), ("target", block["train_end"], block["target_end"])]:
            bounds[block["name"] + "/" + kind] = (timestamp(lower), timestamp(upper), ["future"] if block["name"] == "final" and kind == "target" else ["train", "valid"])
    totals = {name: {"seconds": defaultdict(float), "airings": defaultdict(int)} for name in bounds}
    connection = sqlite3.connect(database.resolve().as_uri()+"?mode=ro", uri=True)
    connection.execute("PRAGMA query_only=ON")
    try:
        channels = [row[0] for row in connection.execute("SELECT DISTINCT channel FROM schedule ORDER BY channel")]
        for channel in channels:
            by_role = defaultdict(list)
            for role, begin, duration, item in connection.execute("SELECT role,start,duration,item_id FROM schedule WHERE channel=? AND duration>0 ORDER BY role,start,item_id,duration", (channel,)):
                scale = audit["files"][role]["duration_scale_seconds"]
                by_role[role].append((begin, begin + duration * scale, item))
            for name, (lower, upper, roles) in bounds.items():
                airings = [row for role in roles for row in by_role[role]]
                seconds, counts = exposure(airings, lower, upper)
                for item, value in seconds.items():totals[name]["seconds"][item] += value
                for item, value in counts.items():totals[name]["airings"][item] += value
    finally:
        connection.close()
    hashes = {}
    for name, values in totals.items():
        lower, upper, roles = bounds[name]
        record = {"lower": lower, "upper": upper, "source_roles": roles, "days": (upper-lower)/86400.,
                  "seconds": dict(values["seconds"]), "airings": dict(values["airings"]),
                  "airing_definition": "Distinct canonical same-item airing on one channel, with >=120 nonconflicting labelled seconds inside window; same-item overlapping rows unioned, adjoining airings separate"}
        path = output / (name.replace("/", "-") + ".json")
        with path.open("x") as stream:json.dump(record,stream,indent=2);stream.write("\n")
        hashes[path.name] = sha256(path)
        if name.endswith("/target"):
            block = name.split("/")[0]
            reference = json.loads((prepared / block / "availability.json").read_text())
            assert all(values["seconds"].get(item,0.) == duration for item,duration in zip(reference["items"],reference["airtime_seconds"]))
    assert sha256(database) == before
    result = {"status":"complete", "created_utc":dt.datetime.now(dt.timezone.utc).isoformat(),
              "wall_seconds":time.perf_counter()-start, "database_sha256_before_and_after":before,
              "database_opened_read_only":True, "view_log_queried":False, "future_content_attributes_read":False,
              "fields_read":["role","channel","start","duration","item_id"],
              "source_sha256":sha256(__file__), "config_sha256":sha256(config_path), "output_hashes":hashes,
              "all_target_seconds_exact_match_existing_audited_availability":True,
              "new_model_quality_metrics_computed":False}
    with (output / "manifest.json").open("x") as stream:json.dump(result,stream,indent=2);stream.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--prepared", type=Path)
    args = parser.parse_args()
    prepare(args.output.resolve(), args.config, args.prepared)
