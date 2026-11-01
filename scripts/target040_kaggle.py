#!/usr/bin/env python3
"""Validate a locally selected candidate and preserve authorized Kaggle evidence.

Read-only inspection is the default. Submission requires --mode submit, an
audited exact CSV, authenticated own-account access, entered competition and a
reported quota. The enclosing user task authorizes designated submissions, not
accepting rules. No credentials or resumable-upload URLs are logged.
"""
import argparse
import csv
import datetime as dt
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sys
import requests

ROOT = Path(__file__).resolve().parents[1]
COMPETITION = "sweettv-tv-program-recommender"


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_run(run):
    manifest, selection, check, audit = [read(run / name) for name in [
        "manifest.json", "selection.json", "final/submission-validation.json", "independent-audit.json"]]
    if manifest["status"] not in {"complete_local_target040_round", "complete_local_v3_evaluation", "complete_local_target040_portfolio_candidate"} or not check["passed"] or not audit["passed"]:
        raise ValueError("Candidate must come from a completed independently audited local run")
    source = run / "final/submission.csv"
    sha = digest(source)
    if sha != check["sha256"] or sha != audit["final_csv_sha256"]:
        raise ValueError("Candidate hash no longer matches audits")
    with Path(manifest["template"]).open(encoding="utf-8-sig", newline="") as stream:
        template = list(csv.reader(stream))
    with source.open(encoding="utf-8-sig", newline="") as stream:
        actual = list(csv.reader(stream))
    if actual[0] != template[0] or len(actual) != 1261 or [r[0] for r in actual[1:]] != [r[0] for r in template[1:]]:
        raise ValueError("Template headers/user strings/order differ")
    pool = set(read(Path(manifest["prepared"]) / "final/candidates.json")["valid"])
    for user, text in actual[1:]:
        tokens = text.split(" ")
        if len(tokens) != 5 or any(not t.isascii() or not t.isdecimal() for t in tokens):
            raise ValueError("Invalid five-ID representation")
        ids = list(map(int, tokens))
        if len(set(ids)) != 5 or min(ids) <= 0 or not set(ids) <= pool or " ".join(map(str, ids)) != text:
            raise ValueError("Invalid programme IDs")
    for name, value in manifest["source_hashes"].items():
        snapshot = Path(manifest.get("source_snapshot_root",run/"source-snapshot")) / name
        implementation = snapshot if snapshot.is_file() else ROOT / name
        if digest(implementation) != value:
            raise ValueError("Implementation/config changed since selected run: " + name)
    return source, sha, selection


def submission_record(item):
    status = getattr(item.status, "name", str(item.status))
    return {"id": int(item.ref), "date": item.date.isoformat() if item.date else None,
        "filename": item.file_name, "description": item.description, "status": status,
        "public_score": item.public_score or None, "private_score": item.private_score or None,
        "total_bytes": item.total_bytes, "submitted_by": item.submitted_by}


def recent_submissions(api):
    """Read unique own history using the live endpoint's supported pageSize."""
    from kagglesdk.competitions.types.competition_api_service import ApiSubmission
    response=requests.get("https://www.kaggle.com/api/v1/competitions/submissions/list/"+COMPETITION,
        params={"pageSize":200},auth=(api.get_config_value("username"),api.get_config_value("key")),timeout=30)
    response.raise_for_status();raw=response.json()
    if not isinstance(raw,list) or len(raw)>=200:
        raise RuntimeError("History reached conservative200-record bound; reconcile before uploading")
    result=[ApiSubmission.from_dict(item) for item in raw]
    if len({int(item.ref) for item in result})!=len(result):
        raise RuntimeError("Repeated submission IDs in own history")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["inspect", "submit", "poll"], default="inspect")
    parser.add_argument("--upload-receipt", type=Path, help="Own prior upload receipt for polling; never resubmits")
    args = parser.parse_args()
    run, destination = args.run_dir.resolve(), args.evidence_dir.resolve()
    if destination.exists():
        raise FileExistsError("Evidence destination must be new")
    source, sha, selection = validate_run(run)
    destination.mkdir(parents=True)
    with (destination / "candidate.csv").open("xb") as stream:
        stream.write(source.read_bytes())
    assert digest(destination / "candidate.csv") == sha
    event = {"created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "mode": args.mode,
        "competition": COMPETITION, "run": str(run), "candidate_sha256": sha, "candidate_bytes": source.stat().st_size,
        "model_sha256": digest(run / "final/model.pkl"), "run_manifest_sha256": digest(run / "manifest.json"),
        "selection_sha256": digest(run / "selection.json"), "frozen_config_sha256": digest(run / "frozen-config.json"),
        "selected_specification": selection["selected_specification"], "rows_validated": 1260,
        "uploaded": False, "success_private_target_verified": False, "status": "validated_locally",
        "rules_accepted_by_tool": False, "performed_by": "Codex account workflow; genuine API observations only"}

    def save(name, record):
        with (destination / name).open("x") as stream:
            json.dump(record, stream, indent=2, ensure_ascii=False);stream.write("\n")
    save("validated-candidate.json", event)
    try:
        # Installed official Kaggle1.7.4.5 auto-authenticates on import.
        from kaggle import api
        username = api.get_config_value("username")
        if not username:
            raise RuntimeError("Authenticated account username is unavailable")
        event["kaggle_username"] = username
        # Closed community competitions may not occur in general search.
        # The authenticated entered list supplies actual account entry/quota.
        entries = api.competitions_list(group="entered", category="all")
        exact = [item for item in entries if item.ref.rstrip("/").split("/")[-1] == COMPETITION]
        if len(exact) != 1:
            raise RuntimeError("Cannot establish competition-specific entry/quota")
        entry = exact[0]
        event.update(user_has_entered=bool(entry.user_has_entered), reported_max_daily_submissions=int(entry.max_daily_submissions))
        if not entry.user_has_entered:
            event["status"] = "user_must_personally_enter_and_accept_rules"
            save("account-observation.json", event)
            return 2
        observed = recent_submissions(api)
        quota_response = requests.get("https://www.kaggle.com/api/v1/competitions/"+COMPETITION+"/submissions/limits",
            auth=(username,api.get_config_value("key")),timeout=30)
        if quota_response.status_code != 200:
            raise RuntimeError("Live submission quota unavailable, HTTP status " + str(quota_response.status_code))
        live_quota = quota_response.json()
        if not isinstance(live_quota.get("numAllowedNow"),int):
            raise RuntimeError("Live quota response lacks an integer allowance")
        event["live_submission_limits"] = live_quota
        save("live-submission-limits.json",live_quota)
        save("submissions-before.json", [submission_record(item) for item in observed])
        threshold = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=1)
        count = sum(item.date is None or (item.date.replace(tzinfo=dt.timezone.utc) if item.date.tzinfo is None else item.date) >= threshold for item in observed)
        event["submissions_within_rolling24h_conservative"] = count
        if args.mode == "inspect":
            event["status"] = "authenticated_read_only_inspection"
        elif args.mode == "submit":
            if live_quota["numAllowedNow"] <= 0:
                event["status"] = "reported_quota_unavailable_or_conservatively_exhausted"
                save("account-observation.json", event)
                return 2
            prior = []
            linkage = ROOT / "evidence/target040-01/historical-submissions/linkage-register.json"
            if linkage.exists():
                prior.extend(f"downloaded-own-submission:{item['submission_id']}" for item in read(linkage)
                             if item.get("downloaded_sha256") == sha)
            for path in (ROOT / "evidence/target040-01").rglob("upload-receipt.json"):
                record = read(path)
                if record.get("candidate_sha256") == sha:
                    prior.append(str(path))
            if prior or any(sha in (item.description or "") for item in observed):
                event.update(status="identical_candidate_previously_submitted_or_in_flight", matching_receipts=prior)
                save("account-observation.json", event)
                return 2
            description = f"Хайрисламов Тимур | Kaggle {username} | target040 {selection['selected_method']} | sha256 {sha}"
            event.update(description=description, status="upload_call_started_no_automatic_retry")
            save("upload-intent.json", event)
            reply = api.competition_submit(str(destination / "candidate.csv"), description, COMPETITION, quiet=True)
            reference = int(getattr(reply, "ref", 0))
            if reference <= 0:
                raise RuntimeError("Upload returned no positive submission reference; inspect account before any retry")
            event.update(uploaded=True, submission_id=reference, status="submitted_awaiting_real_scoring")
            save("upload-receipt.json", event)
        else:
            if args.upload_receipt is None:
                raise ValueError("Polling requires a traceable own upload receipt")
            receipt = read(args.upload_receipt)
            if receipt["candidate_sha256"] != sha or not receipt["uploaded"]:
                raise ValueError("Receipt/candidate mismatch")
            event.update(uploaded=True, submission_id=receipt["submission_id"])
        if event.get("submission_id"):
            matches = [item for item in observed if int(item.ref) == event["submission_id"]]
            if args.mode == "submit":
                matches = [item for item in api.competition_submissions(COMPETITION) if int(item.ref) == event["submission_id"]]
            if matches:
                result = submission_record(matches[0])
                save("real-submission-observation.json", result)
                if sha not in result["description"]:
                    raise ValueError("Returned submission description does not match submitted hash")
                event["real_api_observation"] = result
                event["status"] = result["status"]
                if result["status"] in {"COMPLETE", "SUBMISSION_STATUS_COMPLETE"} and result["private_score"] is not None:
                    score = Decimal(result["private_score"])
                    if not score.is_finite() or not Decimal(0) <= score <= Decimal(1):
                        raise ValueError("Invalid returned private MAP score")
                    event["success_private_target_verified"] = score >= Decimal("0.40000")
        save("account-observation.json", event)
        print(json.dumps({k: event.get(k) for k in ["status", "submission_id", "candidate_sha256", "success_private_target_verified"]}))
        return 0
    except Exception as error:
        # Do not serialize arbitrary SDK exception strings: they can contain URLs.
        event.update(status="account_or_scoring_action_failed", exception_type=type(error).__name__,
                     http_status=getattr(error, "status", None), automatic_retry_performed=False)
        save("account-action-failure.json", event)
        print(json.dumps({"status": event["status"], "exception_type": event["exception_type"], "evidence": str(destination)}))
        return 2


if __name__ == "__main__":
    sys.exit(main())
