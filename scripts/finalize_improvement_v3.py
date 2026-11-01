#!/usr/bin/env python3
"""Validate/copy an audited local candidate and inventory new v3 artifacts.

This command does not publish, upload, alter v2 or edit the report PDF.
"""
import argparse
import csv
import datetime as dt
import hashlib
import io
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    evidence = ROOT / "evidence/improvement-v3-01"
    handoff = evidence / "local-handoff.json"
    inventory = evidence / "artifacts-manifest.json"
    if handoff.exists() or inventory.exists():
        raise FileExistsError("Local handoff/inventory already exists")
    manifest, audit, numerics, selection, validation = [read(run / name) for name in [
        "manifest.json", "independent-audit.json", "independent-ALS-numerics.json", "selection.json", "final/submission-validation.json"]]
    assert manifest["status"] == "complete_local_v3_evaluation"
    assert audit["passed"] and numerics["passed"]
    assert manifest["config_sha256"] == digest(ROOT / "config/improvement-v3.json")
    for name, expected in manifest["source_hashes"].items():
        assert digest(ROOT / name) == expected, name
    source = run / "final/submission.csv"
    data = source.read_bytes()
    assert hashlib.sha256(data).hexdigest() == validation["sha256"] == audit["final_csv_sha256"]
    template_data = Path(manifest["template"]).read_bytes()
    template = list(csv.reader(io.StringIO(template_data.decode("utf-8-sig"))))
    candidate = list(csv.reader(io.StringIO(data.decode("utf-8-sig"))))
    assert candidate[0] == template[0] == ["user_id", "tv_show_id"]
    assert len(candidate) == len(template) == 1261
    assert [r[0] for r in candidate[1:]] == [r[0] for r in template[1:]]
    assert data.startswith(b"\xef\xbb\xbf") == template_data.startswith(b"\xef\xbb\xbf")
    pool = set(read(Path(manifest["prepared"]) / "final/candidates.json")["valid"])
    for user, text in candidate[1:]:
        tokens = text.split(" ")
        assert len(tokens) == 5 and all(token.isascii() and token.isdecimal() for token in tokens)
        ids = list(map(int, tokens))
        assert min(ids) > 0 and len(set(ids)) == 5 and set(ids) <= pool
        assert text == " ".join(map(str, ids))
    destination = ROOT / "submission_v3.csv"
    if destination.exists():
        assert destination.read_bytes() == data, "Refusing to replace a different v3 candidate"
    else:
        with destination.open("xb") as stream:
            stream.write(data)
    assert digest(destination) == validation["sha256"]
    protected = read(evidence / "prechange-manifest.json")["files"]
    for name in ["submission_v2.csv", "evidence/verification.json", "evidence/verification.txt"]:
        assert digest(ROOT / name) == protected[name]["sha256"]
    staging = subprocess.run(["git", "-C", str(ROOT / "repository-staging"), "status", "--porcelain"],
                             capture_output=True, text=True, check=True)
    assert not staging.stdout.strip(), "Public repository staging unexpectedly changed"
    summary = read(run / "summary.json")
    tests = read(evidence / "verification-01/verification.json")
    result = {
        "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scope": "Local v3 only; no Kaggle upload, publication or PDF change",
        "performed_by": "Codex automated local work, not author personal checks",
        "run": str(run.relative_to(ROOT)), "selected_specification": selection["selected_specification"],
        "recommended_replacement_of_v2": selection["recommended_replacement_of_v2"],
        "strict_winning_slices": selection["raw_best_strict_slice_wins_over_v2"],
        "completed_trials": summary["completed_trials"], "failed_trials": summary["failed_trials"],
        "tests_run": tests["tests_run"], "tests_passed": tests["passed"],
        "independent_actual_export_audit": str((run / "independent-audit.json").relative_to(ROOT)),
        "independent_ALS_numerics": str((run / "independent-ALS-numerics.json").relative_to(ROOT)),
        "protected_original_files_unchanged": audit["protected_original_files_unchanged"],
        "protected_metadata_changes": audit["protected_metadata_changes"],
        "legacy_prediction_slots_identical": audit["legacy_prediction_slots_identical"],
        "submission": str(destination), "submission_sha256": digest(destination),
        "headers": candidate[0], "users": 1260, "recommendations_per_user": 5,
        "exact_template_user_strings_order_and_BOM_preserved": True,
        "future_candidate_pool": len(pool), "all_candidate_IDs_positive_valid_distinct": True,
        "report": "docs/improvement-v3-results.md", "staging_clean": True,
        "kaggle_uploaded": False, "kaggle_score": None, "pdf_changed": False}
    handoff.write_text(json.dumps(result, indent=2) + "\n")
    new_sources = [ROOT / name for name in manifest["source_hashes"] if name not in protected]
    scripts = list((ROOT / "scripts").glob("*improvement_v3*.py"))
    paths = set(new_sources + scripts + [destination, ROOT / "docs/improvement-v3-results.md"])
    paths.update(p for p in run.rglob("*") if p.is_file())
    paths.update(p for p in evidence.rglob("*") if p.is_file())
    entries = {str(path.relative_to(ROOT)): {"bytes": path.stat().st_size, "sha256": digest(path)}
               for path in sorted(paths)}
    inventory.write_text(json.dumps({"created_utc": result["completed_utc"], "files": entries,
        "scope": "New v3 code/evidence/run/document/candidate. Prepared inputs are separately hashed in the run manifest. Inventory excludes itself.",
        "external_uploads": False}, indent=2) + "\n")
    print(json.dumps({"passed": True, "selected": selection["selected_method"], "csv_sha256": digest(destination),
                      "files_in_inventory": len(entries), "handoff": str(handoff)}))


if __name__ == "__main__":
    main()
