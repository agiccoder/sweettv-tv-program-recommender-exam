#!/usr/bin/env python3
"""Run all legacy and v3 synthetic tests into a NEW evidence destination."""
import argparse
import datetime as dt
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sweettv.cli import environment
from sweettv.improve_v3 import sources
from verify import RecordedResult


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    if args.destination.exists():
        raise FileExistsError("Verification evidence must not be overwritten")
    args.destination.mkdir(parents=True)
    started = time.perf_counter()
    before = sources()
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    result = unittest.TextTestRunner(stream=stream, verbosity=2, resultclass=RecordedResult).run(suite)
    text = stream.getvalue()
    (args.destination / "verification.txt").write_text(text)
    env = environment()
    env["packages"]["implicit"] = importlib.metadata.version("implicit")
    record = {"scope": "Synthetic correctness fixtures only; no real v3 quality metrics", "performed_by": "Codex automated runner, not author personal checks",
              "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "tests_run": result.testsRun,
              "passed": result.wasSuccessful() and sources() == before, "failures": len(result.failures), "errors": len(result.errors),
              "skipped": len(result.skipped), "wall_seconds": time.perf_counter() - started, "source_hashes": before,
              "source_hashes_stable": sources() == before, "environment": env, "tests": result.outcomes,
              "log_sha256": hashlib.sha256(text.encode()).hexdigest(), "old_verification_overwritten": False}
    (args.destination / "verification.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({key: record[key] for key in ["tests_run", "passed", "failures", "errors", "wall_seconds", "source_hashes_stable"]}))
    if not record["passed"]:
        print(text)
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
