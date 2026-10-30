"""Bounded v2 development/confirmation evaluation of preferred-Top5 proxies.

This creates a new immutable run; original predictions, fitted states and score
evidence are never edited. Selection uses early development primary MAP only.
The later confirmation block is a previously inspected holdout, not a fresh
independent test. No Kaggle interaction or report update occurs in this module.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import datetime as dt
import json
import math
from pathlib import Path
import resource
import sys
import time
import traceback

import numpy as np
import pandas as pd

from .cli import environment, source_hashes, write_csv
from .data import dump_json, sha256
from .metrics import audit_exported_ap, evaluate
from .models import fit_content, fit_popularity, fit_ppr
from .outputs import write_submission


TARGET_KINDS = ("primary", "secondary", "all_observed")


def load_json(path):
    return json.loads(Path(path).read_text())


def distribution(values):
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array):
        return {"count": 0}
    if not np.all(np.isfinite(array)):
        raise ValueError("Nonfinite diagnostic value")
    result = {"count": len(array), "mean": float(array.mean())}
    result.update({label: float(np.quantile(array, q)) for label, q in [
        ("min", 0), ("p10", .1), ("p25", .25), ("median", .5), ("p75", .75), ("p90", .9), ("max", 1)]})
    return result


def read_template(path, expected_users):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if "user_id" not in (reader.fieldnames or []):
            raise ValueError("Template lacks exact user_id header")
        users = [row["user_id"] for row in reader]
    if len(users) != expected_users or len(set(users)) != expected_users or any(not user for user in users):
        raise ValueError("Template user count, uniqueness or null mismatch")
    return users


def prepare_targets(frame, candidates, template_users):
    """Aggregate genuine accepted held-out quantities before selecting Top5.

    Return all three target dictionaries over the same eligible template cohort.
    Tied programme quantities always use programme ID ascending, including ties
    between more than five programmes. The source dataframe is never modified.
    """
    required = {"user_id", "item_id", "seconds", "watches"}
    if not required <= set(frame.columns):
        raise ValueError("Held-out interactions lack user/item/seconds/watches")
    if any(type(user) is not str for user in frame.user_id.unique()):
        raise ValueError("User IDs must remain exact strings")
    quantities = frame[["seconds", "watches"]].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(quantities)) or np.any(quantities < 0):
        raise ValueError("Target quantities must be finite nonnegative")
    grouped = frame.groupby(["user_id", "item_id"], as_index=False, sort=False)[["seconds", "watches"]].sum()
    pool = set(candidates)
    if len(pool) < 5 or any(type(item) is not int or item <= 0 for item in pool):
        raise ValueError("Require at least five positive integer candidates")
    raw = {}
    eligible = {}
    for row in grouped.itertuples(index=False):
        item = int(row.item_id)
        if item <= 0:
            raise ValueError("Unlabelled/nonpositive item cannot be a held-out target")
        record = (item, float(row.seconds), float(row.watches))
        raw.setdefault(row.user_id, []).append(record)
        if item in pool:
            eligible.setdefault(row.user_id, []).append(record)
    targets = {kind: {} for kind in TARGET_KINDS}
    details = []
    cohort = [user for user in template_users if eligible.get(user)]
    if not cohort:
        raise ValueError("No eligible template holdout users")
    for user in template_users:
        records = eligible.get(user, [])
        primary = sorted(records, key=lambda row: (-row[1], row[0]))[:5]
        secondary = sorted(records, key=lambda row: (-row[2], row[0]))[:5]
        all_items = {row[0] for row in records}
        targets["primary"][user] = {row[0] for row in primary}
        targets["secondary"][user] = {row[0] for row in secondary}
        targets["all_observed"][user] = all_items
        total_seconds = math.fsum(row[1] for row in records)
        total_watches = math.fsum(row[2] for row in records)
        intersection = targets["primary"][user] & targets["secondary"][user]
        union = targets["primary"][user] | targets["secondary"][user]
        details.append({
            "user_id": user, "in_cohort": bool(records),
            "primary_ordered_ids": [row[0] for row in primary],
            "secondary_ordered_ids": [row[0] for row in secondary],
            "all_observed_ids": sorted(all_items),
            "primary_item_seconds": {str(row[0]): row[1] for row in primary},
            "secondary_item_watches": {str(row[0]): row[2] for row in secondary},
            "primary_secondary_intersection": len(intersection),
            "primary_secondary_jaccard": len(intersection)/len(union) if union else 0.0,
            "top5_seconds_mass_fraction": math.fsum(row[1] for row in primary)/total_seconds if total_seconds else 0.0,
            "top5_watches_mass_fraction": math.fsum(row[2] for row in secondary)/total_watches if total_watches else 0.0,
            "raw_observed_items": len(raw.get(user, [])),
            "eligible_observed_items": len(all_items),
            "outside_pool_items": len(raw.get(user, []))-len(records),
        })
    included = [row for row in details if row["in_cohort"]]
    audit = {"evaluation_users": len(cohort), "template_users": len(template_users),
             "excluded_no_eligible_targets": len(template_users)-len(cohort),
             "holdout_users_outside_template": len(set(raw)-set(template_users)),
             "candidate_count": len(pool), "user_ids": cohort,
             "target_overlap": {key: distribution(row[key] for row in included) for key in [
                 "eligible_observed_items", "primary_secondary_intersection", "primary_secondary_jaccard",
                 "top5_seconds_mass_fraction", "top5_watches_mass_fraction"]},
             "details": details}
    return cohort, targets, audit


def _validate_prepared(prepared, block):
    protocol = load_json(prepared/"protocol.json")
    bounds = protocol["windows_half_open"]
    expected = [dt.datetime.fromisoformat(block[key]).replace(tzinfo=dt.timezone.utc).timestamp()
                for key in ["train_start", "train_end", "target_end"]]
    if bounds["train"] != expected[:2] or bounds["valid"] != expected[1:]:
        raise ValueError(f"Prepared temporal boundaries disagree with frozen v2 block: {prepared}")
    if protocol["threshold_seconds"] != 120 or protocol["repeat_policy"] != "allow" or protocol["future_views_used"]:
        raise ValueError("Prepared interaction protocol disagrees with v2 freeze")
    return protocol


def fit_method(spec, train, metadata, history_scale=None):
    kind = spec["kind"]
    if kind == "popularity":
        return fit_popularity(train)
    if kind == "content":
        return fit_content(train, metadata)
    if kind == "original_ppr":
        return fit_ppr(train, alpha=spec["alpha"])
    from .improved_models import fit_improved_ppr, fit_personal_history
    if kind == "personal_history":
        return fit_personal_history(train, signal=spec["signal"], power=spec["power"])
    if kind == "improved_ppr":
        return fit_improved_ppr(train, alpha=spec["alpha"], seed_signal=spec["seed_signal"],
                                seed_power=spec["seed_power"], history_scale=history_scale if spec.get("recency") else None)
    raise ValueError(f"Unknown frozen method kind: {kind}")


def read_history_scale(path, train, cutoff):
    """Read keyed, cutoff-specific decayed-seconds features with exact lineage."""
    features = pd.read_parquet(path)
    required = {"user_id", "item_id", "seconds", "recent_seconds"}
    if not required <= set(features.columns):
        raise ValueError(f"Recency features require columns {sorted(required)}")
    if any(type(user) is not str for user in features.user_id.unique()) or features.duplicated(["user_id", "item_id"]).any():
        raise ValueError("Recency keys must be unique with exact string user IDs")
    reference = train[["user_id", "item_id", "seconds"]].groupby(["user_id", "item_id"], as_index=False)["seconds"].sum()
    merged = reference.merge(features, on=["user_id", "item_id"], how="outer", suffixes=("_reference", "_feature"), indicator=True)
    if not (merged["_merge"] == "both").all() or not np.allclose(merged.seconds_reference, merged.seconds_feature, rtol=0, atol=1e-8):
        raise ValueError("Recency features differ from the exact training interaction keys or seconds")
    values = merged.recent_seconds.to_numpy(dtype=np.float64)
    total = merged.seconds_reference.to_numpy(dtype=np.float64)
    if np.any(~np.isfinite(values)) or np.any(values < 0) or np.any(values-total > 1e-7):
        raise ValueError("Decayed seconds must be finite and between zero and historical seconds")
    # A feature-provided end/cutoff column is checked when present; the source
    # feature manifest and input hash remain part of the frozen run evidence.
    expected_cutoff = dt.datetime.fromisoformat(cutoff).replace(tzinfo=dt.timezone.utc).timestamp()
    event_columns = [column for column in ["max_event_end", "max_airing_end", "last_event_end"] if column in features]
    if not event_columns:
        raise ValueError("Recency features must record the last accepted event endpoint")
    for column in event_columns:
        if column in features and (pd.to_numeric(features[column]) > expected_cutoff).any():
            raise ValueError("Recency event timestamp crosses the training cutoff")
        if not np.isfinite(pd.to_numeric(features[column]).to_numpy(dtype=np.float64)).all():
            raise ValueError("Recency event timestamps must be finite")
    scale = np.divide(values, total, out=np.zeros_like(values), where=total>0)
    return {(row.user_id, int(row.item_id)): float(value)
            for row, value in zip(merged.itertuples(index=False), scale)}


def final_refit(prepared, destination, selected, template_path, expected_users, history_scale=None):
    """Refit the development-selected specification on allowed full history."""
    full = pd.read_parquet(prepared/"full-interactions.parquet")
    records = load_json(prepared/"metadata.json")
    metadata = {int(item): dict(record) for item, record in records["train"].items()}
    for item, record in records["valid"].items():
        prior = metadata.setdefault(int(item), {})
        for key, value in record.items():
            if not prior.get(key) and value:
                prior[key] = value
    candidates = load_json(prepared/"candidates.json")["future"]
    users = read_template(template_path, expected_users)
    start = time.perf_counter()
    model = fit_method(selected, full, metadata, history_scale)
    fit_seconds = time.perf_counter()-start
    start = time.perf_counter()
    recommendations = {user: model.recommend(user, candidates) for user in users}
    predict_seconds = time.perf_counter()-start
    ids = {user: [int(row["item_id"]) for row in values] for user, values in recommendations.items()}
    validation = write_submission(template_path, destination/"submission.csv", ids, set(candidates), expected_users)
    validation.update({"selected_method": selected["name"], "selected_from": "early development primary Top5-seconds MAP only",
                       "fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
                       "full_fit_user_item_pairs": len(full), "future_view_features": False})
    model.save(destination/"model.pkl")
    dump_json(destination/"submission-validation.json", validation)
    dump_json(destination/"diagnostics.json", model.diagnostics)
    future_metadata = {int(item): record for item, record in records["future"].items()}
    rows = []
    for user, values in recommendations.items():
        for rank, row in enumerate(values, 1):
            item = int(row["item_id"])
            meta = future_metadata.get(item, {})
            rows.append({"user_id": user, "rank": rank, "item_id": item, "score": float(row["score"]), "reason": row["reason"],
                         "title": meta.get("tv_show_title", ""), "category": meta.get("tv_show_category", "")})
    write_csv(destination/"recommendations.csv", rows, ["user_id", "rank", "item_id", "score", "reason", "title", "category"])
    return validation


def exposure_concentration(recommendations, candidate_count):
    counts = Counter(item for items in recommendations.values() for item in items)
    total = sum(counts.values())
    ranked = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
    return {"distinct_recommended_programmes": len(counts),
            "candidate_coverage_fraction": len(counts)/candidate_count,
            "top5_programmes_exposure_fraction": sum(count for _, count in ranked[:5])/total,
            "exposure_hhi": sum((count/total)**2 for count in counts.values()),
            "most_common_programmes": [{"item_id": item, "recommendation_count": count} for item, count in ranked[:10]],
            "interpretation": "Frequency concentration of returned slots across users; model scores are not comparable probabilities"}


def evaluate_block(prepared, destination, specs, users, config, block_name, history_scale=None):
    destination.mkdir(parents=True, exist_ok=False)
    _validate_prepared(prepared, config[block_name])
    train = pd.read_parquet(prepared/"train-interactions.parquet")
    heldout = pd.read_parquet(prepared/"valid-interactions.parquet")
    candidates = load_json(prepared/"candidates.json")["valid"]
    metadata = {window: {int(item): record for item, record in records.items()}
                for window, records in load_json(prepared/"metadata.json").items()}
    cohort, targets, cohort_audit = prepare_targets(heldout, candidates, users)
    training_users = set(train.user_id)
    warm = [user for user in cohort if user in training_users]
    cold = [user for user in cohort if user not in training_users]
    cohort_audit.update({"warm_users": len(warm), "cold_users": len(cold)})
    dump_json(destination/"cohort-and-targets.json", cohort_audit)
    target_rows = [{"user_id": row["user_id"], "in_cohort": row["in_cohort"],
                    "primary_ids": json.dumps(row["primary_ordered_ids"]),
                    "secondary_ids": json.dumps(row["secondary_ordered_ids"]),
                    "all_observed_ids": json.dumps(row["all_observed_ids"])} for row in cohort_audit["details"]]
    write_csv(destination/"targets.csv", target_rows, ["user_id", "in_cohort", "primary_ids", "secondary_ids", "all_observed_ids"])
    original_cold = None
    methods, all_recommendations = {}, []
    per_user_exports = {kind: [] for kind in TARGET_KINDS}
    for spec in specs:
        name = spec["name"]
        print(json.dumps({"block": block_name, "method": name, "phase": "fit"}), flush=True)
        start = time.perf_counter()
        model = fit_method(spec, train, metadata["train"], history_scale)
        fit_seconds = time.perf_counter()-start
        start = time.perf_counter()
        recommendations = {user: model.recommend(user, candidates) for user in cohort}
        predict_seconds = time.perf_counter()-start
        ids = {user: [int(row["item_id"]) for row in records] for user, records in recommendations.items()}
        for user, items in ids.items():
            if len(items) != len(set(items)) or len(items) != 5 or not set(items) <= set(candidates):
                raise AssertionError(f"Invalid v2 recommendations for {name}/{user}")
        cold_predictions = {user: ids[user] for user in cold}
        if original_cold is None:
            original_cold = cold_predictions
        elif original_cold != cold_predictions:
            raise AssertionError(f"Cold-user fallback differs for {name}")
        metrics = {}
        for kind in TARGET_KINDS:
            result = evaluate(ids, targets[kind], cohort)
            ap = result.pop("per_user_ap")
            result.update({"ap_distribution": distribution(ap.values()), "ap_zero_users": sum(value == 0 for value in ap.values()),
                           "ap_one_users": sum(value == 1 for value in ap.values()),
                           "users_with_hits": sum(value > 0 for value in ap.values()),
                           "warm_map_at_5": math.fsum(ap[user] for user in warm)/len(warm) if warm else None,
                           "cold_map_at_5": math.fsum(ap[user] for user in cold)/len(cold) if cold else None})
            metrics[kind] = result
            per_user_exports[kind].extend({"method": name, "user_id": user, "relevant_ids": json.dumps(sorted(targets[kind][user])),
                                           "predictions": json.dumps(ids[user]), "ap_at_5": ap[user]} for user in cohort)
        for user, records in recommendations.items():
            for rank, record in enumerate(records, 1):
                item = int(record["item_id"])
                meta = metadata["valid"].get(item, {})
                all_recommendations.append({"method": name, "user_id": user, "rank": rank, "item_id": item,
                                            "score": float(record["score"]), "reason": record["reason"],
                                            "title": meta.get("tv_show_title", ""), "category": meta.get("tv_show_category", ""),
                                            "primary_hit": item in targets["primary"][user], "secondary_hit": item in targets["secondary"][user],
                                            "all_observed_hit": item in targets["all_observed"][user]})
        methods[name] = {"specification": spec, "fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
                         "metrics": metrics, "score_concentration": exposure_concentration(ids, len(candidates))}
        model.save(destination/"models"/f"{name}.pkl")
        dump_json(destination/"diagnostics"/f"{name}.json", model.diagnostics)
        print(json.dumps({"block": block_name, "method": name, "phase": "complete",
                          "primary_map_at_5": metrics["primary"]["map_at_5"],
                          "fit_seconds": fit_seconds, "predict_seconds": predict_seconds}), flush=True)
    write_csv(destination/"recommendations.csv", all_recommendations,
              ["method", "user_id", "rank", "item_id", "score", "reason", "title", "category", "primary_hit", "secondary_hit", "all_observed_hit"])
    audits = {}
    for kind, records in per_user_exports.items():
        path = destination/f"per-user-ap-{kind}.csv"
        write_csv(path, records, ["method", "user_id", "relevant_ids", "predictions", "ap_at_5"])
        audits[kind] = audit_exported_ap(path, expected_map={name: values["metrics"][kind]["map_at_5"] for name, values in methods.items()})
        dump_json(destination/f"independent-ap-audit-{kind}.json", audits[kind])
    result = {"block": block_name, "candidate_count": len(candidates), "cohort_size": len(cohort),
              "warm_users": len(warm), "cold_users": len(cold), "methods": methods,
              "all_exact_fraction_audits_passed": all(audit["passed"] for audit in audits.values()),
              "model_scores_are_probabilities": False}
    dump_json(destination/"summary.json", result)
    return result


def run(args):
    config_path = Path(args.config).resolve()
    config = load_json(config_path)
    target = Path(args.run_dir).resolve()
    if target.exists():
        raise FileExistsError("Immutable v2 run directory already exists")
    target.mkdir(parents=True)
    started = time.perf_counter()
    manifest = {"status": "running", "protocol_id": config["protocol_id"], "config_sha256": sha256(config_path),
                "config_frozen_before_metrics": config, "source_hashes": source_hashes(), "environment": environment(),
                "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "synthetic_fixture": bool(args.synthetic_fixture),
                "external_uploads": False, "original_results_modified": False}
    dump_json(target/"frozen-config.json", config)
    dump_json(target/"manifest.json", manifest)
    try:
        users = read_template(args.template, args.expected_users)
        prepared_paths = {"development": Path(args.development_prepared).resolve(), "confirmation": Path(args.confirmation_prepared).resolve()}
        input_hashes = {}
        for block, prepared in prepared_paths.items():
            for name in ["train-interactions.parquet", "valid-interactions.parquet", "candidates.json", "metadata.json", "protocol.json"]:
                input_hashes[f"{block}/{name}"] = sha256(prepared/name)
        input_hashes["full/full-interactions.parquet"] = sha256(prepared_paths["confirmation"]/"full-interactions.parquet")
        recency_paths = {}
        if any(spec.get("recency") for spec in config["methods"]):
            if not args.recency_dir:
                raise ValueError("Frozen recency variant requires --recency-dir")
            recency_dir = Path(args.recency_dir).resolve()
            recency_paths = {block: recency_dir/name for block, name in [
                ("development", "development_train.parquet"), ("confirmation", "confirmation_train.parquet"), ("full", "full.parquet")]}
            for block, path in recency_paths.items():
                input_hashes[f"recency/{block}"] = sha256(path)
            for path in sorted(recency_dir.glob("*.json")):
                input_hashes[f"recency-evidence/{path.name}"] = sha256(path)
        input_hashes["template"] = sha256(args.template)
        manifest["input_hashes"] = input_hashes
        dump_json(target/"manifest.json", manifest)
        development_scale = read_history_scale(recency_paths["development"], pd.read_parquet(prepared_paths["development"]/"train-interactions.parquet"), config["development"]["train_end"]) if recency_paths else None
        development = evaluate_block(prepared_paths["development"], target/"development", config["methods"], users, config, "development", development_scale)
        eligible = [spec for spec in config["methods"] if spec["selection_eligible"]]
        selected = max(eligible, key=lambda spec: development["methods"][spec["name"]]["metrics"]["primary"]["map_at_5"])
        selection = {"selected_method": selected["name"], "selected_specification": selected,
                     "selection_source": "Early development primary Top5-seconds MAP@5 only",
                     "confirmation_metrics_available_at_selection": False,
                     "eligible_primary_map": {spec["name"]: development["methods"][spec["name"]]["metrics"]["primary"]["map_at_5"] for spec in eligible},
                     "tie_policy": "First method in frozen configuration array wins exact MAP ties",
                     "selected_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
        dump_json(target/"selection.json", selection)
        confirmation_specs = [spec for spec in config["methods"] if spec["kind"] in {"popularity", "content", "original_ppr"}]
        if selected["name"] not in {spec["name"] for spec in confirmation_specs}:
            confirmation_specs.append(selected)
        confirmation_scale = read_history_scale(recency_paths["confirmation"], pd.read_parquet(prepared_paths["confirmation"]/"train-interactions.parquet"), config["confirmation"]["train_end"]) if recency_paths and selected.get("recency") else None
        confirmation = evaluate_block(prepared_paths["confirmation"], target/"confirmation", confirmation_specs, users, config, "confirmation", confirmation_scale)
        full_scale = read_history_scale(recency_paths["full"], pd.read_parquet(prepared_paths["confirmation"]/"full-interactions.parquet"), config["confirmation"]["target_end"]) if recency_paths and selected.get("recency") else None
        final_validation = final_refit(prepared_paths["confirmation"], target/"final", selected, args.template, args.expected_users, full_scale)
        if source_hashes() != manifest["source_hashes"]:
            raise RuntimeError("Source hashes changed during the v2 run; refusing a successful run status")
        final = {"status": "complete_synthetic_fixture" if args.synthetic_fixture else "complete_local_v2_proxy_evaluation",
                 "selected_method": selected["name"], "selection_frozen_before_confirmation": True,
                 "primary_target": config["targets"]["primary"], "hidden_evaluator_identity_established": False,
                 "original_confirmation_holdout_previously_inspected": True,
                 "development": {name: values["metrics"]["primary"]["map_at_5"] for name, values in development["methods"].items()},
                 "confirmation": {name: values["metrics"]["primary"]["map_at_5"] for name, values in confirmation["methods"].items()},
                 "final_submission_sha256": final_validation["sha256"], "external_uploads": False}
        dump_json(target/"summary.json", final)
        manifest["status"] = final["status"]
    except Exception as error:
        manifest["status"] = "failed"
        dump_json(target/"failure.json", {"exception_type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc(),
                                         "external_uploads": False, "original_results_modified": False})
        raise
    finally:
        manifest["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        manifest["wall_seconds"] = time.perf_counter()-started
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        manifest["peak_resident_memory_bytes"] = peak if sys.platform == "darwin" else peak*1024
        manifest["source_hashes_at_end"] = source_hashes()
        manifest["source_changed_during_run"] = manifest["source_hashes_at_end"] != manifest["source_hashes"]
        dump_json(target/"manifest.json", manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-prepared", required=True)
    parser.add_argument("--confirmation-prepared", required=True)
    parser.add_argument("--template", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--config", default="config/improvement-v2.json")
    parser.add_argument("--expected-users", type=int, default=1260)
    parser.add_argument("--synthetic-fixture", action="store_true")
    parser.add_argument("--recency-dir", help="Frozen development_train, confirmation_train and full recency Parquet features")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
