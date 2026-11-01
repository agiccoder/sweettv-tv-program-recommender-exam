"""Execute a frozen target040 round, preserving v3 and requiring external proof for success."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import datetime as dt
import gzip
import importlib.metadata
from fractions import Fraction
import json
import math
from pathlib import Path
import pickle
import resource
import shutil
import sys
import time
import traceback

import numpy as np
import pandas as pd
from scipy import sparse

from .cli import environment, source_hashes
from .data import dump_json, sha256, timestamp
from .improve import TARGET_KINDS, exposure_concentration, read_template
from .metrics import ap_at_k, audit_exported_ap, evaluate
from .outputs import write_submission
from .v3_context import ContextState
from .v3_models import SelectedModel
from .target040_models import Family, required_components

ROOT = Path(__file__).resolve().parents[2]


def load(path):
    return json.loads(Path(path).read_text())


def sources():
    result = source_hashes()
    for name in ["requirements-v3.lock", "scripts/prepare_improvement_v3.py", "scripts/prepare_target040_exposure.py", "scripts/prepare_target040_engagement.py", "scripts/prepare_target040_schedule_factors.py"]:
        result[name] = sha256(ROOT / name)
    return result


def peak_rss():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == "darwin" else value * 1024


def resource_guard(config, elapsed=None):
    if peak_rss() > config["resources"]["max_peak_rss_bytes"]:
        raise MemoryError("Measured process peak RSS exceeds frozen bound")
    if elapsed is not None and elapsed > config["resources"]["max_total_evaluation_seconds"]:
        raise TimeoutError("V3 evaluation exceeds frozen total wall-time bound")


def writer(path, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = gzip.open(path, "wt", newline="", encoding="utf-8") if path.suffix == ".gz" else path.open("w", newline="", encoding="utf-8")
    handle = csv.DictWriter(stream, fieldnames=fields)
    handle.writeheader()
    return stream, handle


def read_recency(path, train, cutoff):
    frame = pd.read_parquet(path)
    keys = ["user_id", "item_id", "seconds", "watches"]
    pd.testing.assert_frame_equal(frame[keys].sort_values(keys[:2]).reset_index(drop=True),
        train[keys].sort_values(keys[:2]).reset_index(drop=True), check_dtype=False, check_exact=True)
    if frame.duplicated(keys[:2]).any() or not frame.user_id.map(lambda user: isinstance(user, str)).all():
        raise ValueError("Recency keys or exact user strings differ")
    if np.any(~np.isfinite(frame.recent_seconds)) or (frame.recent_seconds < 0).any() or (frame.recent_seconds > frame.seconds * (1 + 1e-14)).any():
        raise ValueError("Recency values invalid")
    if np.any(~np.isfinite(frame.last_event_end)) or (frame.last_event_end > cutoff).any():
        raise ValueError("Training recency crosses cutoff")
    return frame


def read_slice(prepared, block, config, final=False):
    directory = prepared / ("final" if final else block["name"])
    train = pd.read_parquet(directory / "train-interactions.parquet")
    recency = read_recency(prepared / "recency" / (("full" if final else block["name"]) + ".parquet"), train, timestamp(block["train_end"]))
    state = ContextState.load(prepared / "context" / ("full" if final else block["name"]))
    if state.cutoff != timestamp(block["train_end"]) or state.lower != timestamp(block["train_start"]):
        raise ValueError("Context cutoff differs from frozen slice")
    if not final:
        protocol = load(directory / "protocol.json")
        if protocol["windows_half_open"] != {"train": [timestamp(block["train_start"]), timestamp(block["train_end"])],
                                              "valid": [timestamp(block["train_end"]), timestamp(block["target_end"])]}:
            raise ValueError("Interaction windows differ from frozen slice")
        if protocol["future_views_used"] or protocol["threshold_seconds"] != config["threshold_seconds"]:
            raise ValueError("Unexpected interaction protocol")
    metadata = {role: {int(item): record for item, record in values.items()}
                for role, values in load(directory / "metadata.json").items()}
    items = load(directory / "candidates.json")["valid"]
    availability = load(directory / "availability.json")
    if items != availability["items"] or len(items) < 5:
        raise ValueError("Availability IDs must match the complete common candidate pool")
    exposure = sparse.load_npz(directory / "availability.npz")
    if exposure.shape != (len(items), state.counts.shape[1]):
        raise ValueError("Availability matrix has wrong training-channel vocabulary")
    return directory, train, recency, state, exposure, items, metadata


def segment_users(cohort, train, context):
    lengths = train.groupby("user_id").item_id.nunique().to_dict()
    result = {name: [] for name in ["short", "medium", "long", "channel_only", "no_history"]}
    for user in cohort:
        n = lengths.get(user, 0)
        raw = user in context.user_index and context.totals[context.user_index[user]] > 0
        name = "channel_only" if n == 0 and raw else "no_history" if n == 0 else "short" if n <= 20 else "medium" if n < 100 else "long"
        result[name].append(user)
    return result, lengths


def select_strategy(summaries, config):
    order = config["selection"]["tie_order"]
    complete = [spec for spec in config["methods"] if all(spec["name"] in summary["methods"] for summary in summaries.values())]
    criteria, exact_criteria, per_slice, exact_slices = {}, {}, {}, {}
    for spec in complete:
        name = spec["name"]
        values = {}
        for block, result in summaries.items():
            metric = result["methods"][name]["metrics"]
            fractions = [Fraction(metric[kind]["exact_macro_mean"]) if "exact_macro_mean" in metric[kind]
                         else Fraction(str(metric[kind]["map_at_5"])) for kind in ["primary", "secondary"]]
            values[block] = sum(fractions, Fraction(0)) / 2
        exact_slices[name] = values
        per_slice[name] = {key: float(value) for key, value in values.items()}
        exact_criteria[name] = sum(values.values(), Fraction(0)) / len(config["slices"])
        criteria[name] = float(exact_criteria[name])
    if config["reference_method"] not in criteria:
        raise RuntimeError("Previous-best reference must finish all slices")
    ranked = sorted(criteria, key=lambda name: (-exact_criteria[name], order.index(name)))
    proposed = ranked[0]
    wins = sum(exact_slices[proposed][block] > exact_slices[config["reference_method"]][block] for block in summaries)
    replace = proposed != config["reference_method"] and exact_criteria[proposed] > exact_criteria[config["reference_method"]] and wins >= 2
    selected = proposed if replace else config["reference_method"]
    spec = next(spec for spec in config["methods"] if spec["name"] == selected)
    return {"selected_method": selected, "selected_specification": spec, "raw_best_method": proposed,
            "recommended_replacement_of_previous_best": replace, "aggregate_criterion": criteria, "per_slice_criterion": per_slice,
            "exact_aggregate_criterion": {key: str(value) for key, value in exact_criteria.items()},
            "exact_per_slice_criterion": {name: {key: str(value) for key, value in values.items()} for name, values in exact_slices.items()},
            "ranked_complete_methods": ranked, "raw_best_strict_slice_wins_over_reference": wins,
            "excluded_incomplete_methods": [spec["name"] for spec in config["methods"] if spec["name"] not in criteria],
            "selection_rule": config["selection"], "selection_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "final_refit_started_at_selection": False, "kaggle_scores_used": False}


def target_subsets(targets, predictions, train_pairs, global_items):
    result = {}
    for label, predicate in [("seen_by_same_user", lambda user, item: (user, item) in train_pairs),
                             ("absent_from_all_train", lambda user, item: item not in global_items),
                             ("known_but_not_seen_by_user", lambda user, item: item in global_items and (user, item) not in train_pairs)]:
        positions, hits, n_users = 0, 0, 0
        for user, relevant in targets.items():
            subset = {item for item in relevant if predicate(user, item)}
            positions += len(subset)
            hits += len(subset & set(predictions[user]))
            n_users += bool(subset)
        result[label] = {"target_positions": positions, "hits_at_5": hits, "users_with_subset": n_users,
                         "micro_recall_at_5": hits / positions if positions else None, "small_denominator": n_users < 30}
    return result


def evaluate_slice(prepared, target, block, config, started):
    target.mkdir(parents=True, exist_ok=False)
    directory, train, recency, context, exposure, candidates, metadata = read_slice(prepared, block, config)
    frozen = load(directory / "cohort-and-targets.json")
    if not frozen["frozen_before_model_scoring"]:
        raise ValueError("Evaluation cohort was not frozen")
    cohort = frozen["user_ids"]
    targets = {kind: {} for kind in TARGET_KINDS}
    for row in frozen["details"]:
        for kind, key in [("primary", "primary_ordered_ids"), ("secondary", "secondary_ordered_ids"), ("all_observed", "all_observed_ids")]:
            targets[kind][row["user_id"]] = set(row[key])
    targets = {kind: {user: values[user] for user in cohort} for kind, values in targets.items()}
    shutil.copy2(directory / "cohort-and-targets.json", target / "cohort-and-targets.json")
    fit_start = time.perf_counter()
    family = Family(train, metadata["train"], recency, context, exposure, candidates, config, config["methods"])
    family_fit_seconds = time.perf_counter() - fit_start
    resource_guard(config, time.perf_counter() - started)
    if family.failures:
        dump_json(target / "component-failures.json", family.failures)
    segments, lengths = segment_users(cohort, train, context)
    dump_json(target / "segments.json", {"definitions": config["segments"], "users": segments,
                                         "counts": {name: len(values) for name, values in segments.items()}})
    extra = [user for user in config["examples"] if user not in cohort]
    users = cohort + extra
    native_stream, native_writer = writer(target / "native-scores.csv.gz", ["component", "user_id", "rank", "item_id", "score", "popularity_padding"])
    cache = {}
    native_start = time.perf_counter()
    try:
        for user in users:
            cache[user] = family.lists(user)
            for name, rows in cache[user].items():
                for rank, (item, score) in enumerate(rows, 1):
                    native_writer.writerow({"component": name, "user_id": user, "rank": rank,
                                            "item_id": item, "score": score, "popularity_padding": False})
    finally:
        native_stream.close()
    native_wall_seconds = time.perf_counter() - native_start
    dump_json(target / "native-score-diagnostics.json", family.native_diagnostics)
    prediction_stream, prediction_writer = writer(target / "recommendations.csv", ["method", "user_id", "in_cohort", "rank", "item_id", "score", "reason", "title", "category", "primary_hit", "secondary_hit", "all_observed_hit", "contributions"])
    contribution_stream, contribution_writer = writer(target / "fusion-contributions.csv.gz", ["method", "user_id", "item_id", "component", "rank", "weight", "native_score", "contribution"])
    union_stream, union_writer = writer(target / "candidate-unions.csv.gz", ["method", "user_id", "native_ids", "fallback_ids", "primary_native_recall", "secondary_native_recall", "primary_with_fallback_recall", "secondary_with_fallback_recall"])
    ap_streams, ap_writers = {}, {}
    for kind in TARGET_KINDS:
        ap_streams[kind], ap_writers[kind] = writer(target / ("per-user-ap-" + kind + ".csv"), ["method", "user_id", "relevant_ids", "predictions", "ap_at_5"])
    train_pairs = set(zip(train.user_id, train.item_id))
    global_items = set(int(item) for item in train.item_id)
    methods, failures, examples = {}, {}, {}
    primary_cold = None
    try:
        for spec in config["methods"]:
            name = spec["name"]
            deps = required_components([spec], config["methods"])
            if deps - set(family.models):
                failures[name] = {"status": "unavailable_dependency", "components": sorted(deps - set(family.models)), "cause": family.failures}
                dump_json(target / "trials" / (name + ".json"), {"specification": spec, **failures[name]})
                continue
            predictions, records_by_user = {}, {}
            recalls = {kind: {"native": [], "with_fallback": [], "positions": 0, "native_hits": 0, "augmented_hits": 0} for kind in ["primary", "secondary"]}
            fallback_slots, start = 0, time.perf_counter()
            for user in users:
                records, contributions, union = family.recommend_from(user, spec, cache[user])
                ids = [int(row["item_id"]) for row in records]
                if len(set(ids)) != 5 or not set(ids) <= family.pool or any(not math.isfinite(row["score"]) for row in records):
                    raise AssertionError("Invalid recommendations")
                records_by_user[user] = records
                if user in targets["primary"]:
                    predictions[user] = ids
                    fallback = {row["item_id"] for row in records if "training_popularity" in row["reason"]}
                    fallback_slots += len(fallback)
                    values = {}
                    for kind in ["primary", "secondary"]:
                        relevant = targets[kind][user]
                        native_hits, augmented_hits = len(relevant & union), len(relevant & (union | fallback))
                        recalls[kind]["native"].append(native_hits / len(relevant))
                        recalls[kind]["with_fallback"].append(augmented_hits / len(relevant))
                        recalls[kind]["positions"] += len(relevant)
                        recalls[kind]["native_hits"] += native_hits
                        recalls[kind]["augmented_hits"] += augmented_hits
                        values[kind + "_native_recall"] = native_hits / len(relevant)
                        values[kind + "_with_fallback_recall"] = augmented_hits / len(relevant)
                    union_writer.writerow({"method": name, "user_id": user, "native_ids": json.dumps(sorted(union)),
                                           "fallback_ids": json.dumps(sorted(fallback)), **values})
                for item, by_component in contributions.items():
                    for component, values in by_component.items():
                        contribution_writer.writerow({"method": name, "user_id": user, "item_id": item,
                                                      "component": component, **values})
                for rank, row in enumerate(records, 1):
                    item = row["item_id"]
                    meta = metadata["valid"].get(item, {})
                    prediction_writer.writerow({"method": name, "user_id": user, "in_cohort": user in cohort,
                        "rank": rank, **row, "title": meta.get("tv_show_title", ""), "category": meta.get("tv_show_category", ""),
                        "primary_hit": item in targets["primary"].get(user, set()), "secondary_hit": item in targets["secondary"].get(user, set()),
                        "all_observed_hit": item in targets["all_observed"].get(user, set()), "contributions": json.dumps(contributions.get(item, {}))})
            composition_seconds = time.perf_counter() - start
            metrics, ap_values = {}, {}
            for kind in TARGET_KINDS:
                result = evaluate(predictions, targets[kind], cohort)
                per_user = result.pop("per_user_ap")
                ap_values[kind] = per_user
                result["segments"] = {segment: {"users": len(ids), "small_denominator": len(ids) < 30,
                    "map_at_5": math.fsum(per_user[user] for user in ids) / len(ids) if ids else None} for segment, ids in segments.items()}
                if kind != "all_observed":
                    result["target_subsets"] = target_subsets(targets[kind], predictions, train_pairs, global_items)
                    if any(ap_values[kind][user] > recalls[kind]["with_fallback"][i] + 1e-12 for i, user in enumerate(cohort)):
                        raise AssertionError("Final AP exceeds retrieval-plus-fallback upper bound")
                metrics[kind] = result
                for user in cohort:
                    ap_writers[kind].writerow({"method": name, "user_id": user,
                        "relevant_ids": json.dumps(sorted(targets[kind][user])), "predictions": json.dumps(predictions[user]), "ap_at_5": per_user[user]})
            cold = {user: predictions[user] for user in segments["no_history"]}
            if primary_cold is None:
                primary_cold = cold
            elif primary_cold != cold:
                raise AssertionError("Truly cold fallback differs across methods")
            candidate_recall = {}
            for kind, values in recalls.items():
                candidate_recall[kind] = {"macro_native_union_recall": math.fsum(values["native"]) / len(cohort),
                    "macro_union_plus_actual_fallback_recall": math.fsum(values["with_fallback"]) / len(cohort),
                    "target_positions": values["positions"], "native_union_hits": values["native_hits"],
                    "union_plus_fallback_hits": values["augmented_hits"], "micro_native_recall": values["native_hits"] / values["positions"],
                    "micro_union_plus_fallback_recall": values["augmented_hits"] / values["positions"],
                    "interpretation": "Native retrieval precedes common fallback; augmented union bounds final AP for these at-most5 target sets"}
            fit_deps = math.fsum(family.fit_seconds[dep] for dep in deps)
            component_inference = math.fsum(family.predict_seconds[dep] for dep in deps)
            result = {"specification": spec, "metrics": metrics, "candidate_recall": candidate_recall,
                "fit_seconds_cached_components": fit_deps, "shared_data_fit_seconds": family.shared_data_seconds,
                "shared_original_graph_fit_seconds": family.shared_graph_seconds if deps & {"ppr_v2", "original_ppr_0.5"} else 0.,
                "predict_seconds_cached_components": component_inference, "ranking_export_seconds": composition_seconds,
                "timing_scope": "New-round measured component/shared fit and prediction/export times; same-cutoff reused fits are zero new work. Shared costs are not additive across trials.",
                "process_highwater_rss_bytes_so_far": peak_rss(), "fallback_slots": fallback_slots,
                "coverage": exposure_concentration(predictions, len(candidates))}
            methods[name] = result
            dump_json(target / "trials" / (name + ".json"), {"status": "complete", **result})
            for user in config["examples"]:
                examples.setdefault(user, {"in_cohort": user in cohort, "methods": {}})["methods"][name] = [
                    {**row, "rank": rank, "title": metadata["valid"].get(row["item_id"], {}).get("tv_show_title", ""),
                     "category": metadata["valid"].get(row["item_id"], {}).get("tv_show_category", ""),
                     "primary_hit": row["item_id"] in targets["primary"].get(user, set()),
                     "secondary_hit": row["item_id"] in targets["secondary"].get(user, set())}
                    for rank, row in enumerate(records_by_user[user], 1)]
            print(json.dumps({"slice": block["name"], "method": name, "primary": metrics["primary"]["map_at_5"],
                              "secondary": metrics["secondary"]["map_at_5"]}), flush=True)
            resource_guard(config, time.perf_counter() - started)
    finally:
        for stream in [prediction_stream, contribution_stream, union_stream, *ap_streams.values()]:
            stream.close()
    audits = {}
    for kind in TARGET_KINDS:
        audits[kind] = audit_exported_ap(target / ("per-user-ap-" + kind + ".csv"), expected_map={name: result["metrics"][kind]["map_at_5"] for name, result in methods.items()})
        for name, values in audits[kind]["methods"].items():
            methods[name]["metrics"][kind]["exact_macro_mean"] = str(Fraction(values["exact_numerator"], values["exact_denominator"]))
        dump_json(target / ("independent-AP-audit-" + kind + ".json"), audits[kind])
    heldout = pd.read_parquet(directory / "valid-interactions.parquet")
    for user, values in examples.items():
        for label, frame, role in [("training_history", train, "train"), ("heldout_watching", heldout, "valid")]:
            rows = []
            for row in frame[frame.user_id == user].sort_values(["seconds", "item_id"], ascending=[False, True]).to_dict("records"):
                item = int(row["item_id"])
                rows.append({**row, "title": metadata[role].get(item, {}).get("tv_show_title", ""),
                             "category": metadata[role].get(item, {}).get("tv_show_category", "")})
            values[label] = rows
        values["training_item_count"] = int(lengths.get(user, 0))
        values["raw_context_seconds"] = float(context.totals[context.user_index[user]]) if user in context.user_index else 0.
        values["top5_targets"] = {kind: sorted(targets[kind].get(user, set())) for kind in ["primary", "secondary"]}
    dump_json(target / "two-user-examples.json", examples)
    for name, model in family.models.items():
        dump_json(target / "diagnostics" / (name + ".json"), model.diagnostics)
        if name.startswith("als"):
            (target / "models").mkdir(exist_ok=True)
            model.model.save(target / "models" / (name + ".npz"))
    with (target / "fitted-components.pkl").open("wb") as stream:
        pickle.dump(family, stream, protocol=pickle.HIGHEST_PROTOCOL)
    result = {"slice": block, "cohort_size": len(cohort), "candidate_count": len(candidates),
              "segments": {key: len(ids) for key, ids in segments.items()}, "methods": methods, "failures": failures,
              "family_fit_wall_seconds": family_fit_seconds, "native_extraction_wall_seconds": native_wall_seconds,
              "shared_data_seconds": family.shared_data_seconds, "shared_graph_seconds": family.shared_graph_seconds,
              "peak_process_rss_bytes": peak_rss(), "exact_AP_audits_passed": all(v["passed"] for v in audits.values())}
    dump_json(target / "summary.json", result)
    return result


def paired_bootstrap(target, selection, config):
    names = [selection["selected_method"]]
    competitors = [name for name in selection["ranked_complete_methods"] if name != config["reference_method"]]
    if competitors:
        names.append(competitors[0])
    by_user = {}
    for slice_index, block in enumerate(config["slices"]):
        values = {}
        for kind in ["primary", "secondary"]:
            with (target / block["name"] / ("per-user-ap-" + kind + ".csv")).open(newline="") as stream:
                for row in csv.DictReader(stream):
                    if row["method"] in set(names + [config["reference_method"]]):
                        values.setdefault(row["user_id"], {}).setdefault(row["method"], 0.)
                        values[row["user_id"]][row["method"]] += .5 * float(row["ap_at_5"])
        by_user[block["name"]] = values
    users = sorted(set().union(*(set(values) for values in by_user.values())))
    result = {}
    for name in set(names):
        array = np.full((len(users), len(config["slices"])), np.nan)
        for j, block in enumerate(config["slices"]):
            for i, user in enumerate(users):
                if user in by_user[block["name"]]:
                    values = by_user[block["name"]][user]
                    array[i, j] = values[name] - values[config["reference_method"]]
        rng = np.random.default_rng(config["bootstrap"]["seed"])
        finite = np.isfinite(array)
        values = np.nan_to_num(array)
        samples = []
        for _ in range(config["bootstrap"]["resamples"]):
            multiplicity = np.bincount(rng.integers(len(users), size=len(users)), minlength=len(users))
            samples.append(float(np.mean((multiplicity @ values) / (multiplicity @ finite))))
        result[name] = {"observed_delta": float(np.nanmean(array, axis=0).mean()),
                       "percentile_interval": np.quantile(samples, config["bootstrap"]["interval"]).tolist()}
    dump_json(target / "post-selection-bootstrap.json", {"config": config["bootstrap"], "comparisons_vs_reference": result,
              "warning": "Post-selection descriptive interval; same user multiplicity across overlapping slices. Does not correct model-selection bias or prove future Kaggle improvement."})


def refit_final(prepared, target, config, selection, template, expected_users):
    target.mkdir(parents=True)
    block = config["final"]
    _, train, recency, context, exposure, candidates, metadata = read_slice(prepared, block, config, final=True)
    spec = selection["selected_specification"]
    fit_start = time.perf_counter()
    family = Family(train, metadata["train"], recency, context, exposure, candidates, config, [spec])
    if family.failures:
        raise RuntimeError("Selected final method failed: " + json.dumps(family.failures))
    fit_seconds = time.perf_counter() - fit_start
    selected = SelectedModel(family, spec)
    users = read_template(template, expected_users)
    stream, handle = writer(target / "recommendations.csv", ["user_id", "rank", "item_id", "score", "reason", "contributions"])
    predictions, examples = {}, {}
    start = time.perf_counter()
    try:
        for user in users:
            records, contributions, _ = family.recommend_from(user, spec, family.lists(user))
            predictions[user] = [row["item_id"] for row in records]
            for rank, row in enumerate(records, 1):
                handle.writerow({"user_id": user, "rank": rank, **row, "contributions": json.dumps(contributions.get(row["item_id"], {}))})
            if user in set(users[:3] + config["examples"]):
                examples[user] = records
    finally:
        stream.close()
    predict_seconds = time.perf_counter() - start
    validation = write_submission(template, target / "submission.csv", predictions, set(candidates), expected_users)
    selected.save(target / "model.pkl")
    with (target / "model.pkl").open("rb") as stream:
        restored = pickle.load(stream)
    for user, records in examples.items():
        if restored.recommend(user) != records:
            raise AssertionError("Final selected model persistence changed predictions")
    for name, model in family.models.items():
        if name.startswith("als"):
            model.model.save(target / (name + ".npz"))
    validation.update({"selected_specification": spec, "fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
                       "training_pairs": len(train), "refit_window": block, "model_roundtrip_users": list(examples),
                       "recommended_replacement_of_previous_best": selection["recommended_replacement_of_previous_best"],
                       "kaggle_uploaded": False, "kaggle_score": None, "future_content_attributes_used": False})
    dump_json(target / "submission-validation.json", validation)
    return validation


def run(args):
    config_path, prepared, target = Path(args.config).resolve(), Path(args.prepared).resolve(), Path(args.run_dir).resolve()
    if target.exists():
        raise FileExistsError("V3 run must be immutable and new")
    config = load(config_path)
    if not args.synthetic_fixture:
        freeze = load(ROOT / config["freeze_evidence"])
        if sha256(config_path) != freeze["sha256"]:
            raise ValueError("Config differs from pre-metric v3 freeze")
    if len(config["methods"]) > config["resources"]["max_configurations_per_slice"]:
        raise ValueError("Frozen method budget exceeded")
    preparation = load(prepared / "preparation-manifest.json")
    if preparation["status"] != "complete" or preparation["config_sha256"] != config["previous_config_sha256"]:
        raise ValueError("Prepared inputs do not identify completed current frozen config")
    cached_manifest = load(ROOT / config["cache_run"] / "manifest.json")
    if "rate_features_root" in config:
        for name, value in config["rate_features_hashes"].items():
            if sha256(ROOT / config["rate_features_root"] / name) != value:
                raise ValueError("Frozen exposure feature changed: " + name)
    for name, value in config.get("engagement", {}).get("hashes", {}).items():
        if sha256(ROOT / config["engagement"]["root"] / name) != value:
            raise ValueError("Frozen engagement feature changed: " + name)
    for name,value in config.get("inductive_mf",{}).get("hashes",{}).items():
        if sha256(ROOT/config["inductive_mf"]["root"]/name) != value:
            raise ValueError("Frozen schedule descriptor changed: " + name)
    if not args.synthetic_fixture:
        if cached_manifest["status"] not in {"complete_local_v3_evaluation", "complete_local_target040_round"}:
            raise ValueError("Cached reference run did not complete")
        for name, value in (cached_manifest["input_hashes"] if config.get("reuse_cached_fits", True) else {}).items():
            if sha256(prepared / name) != value:
                raise ValueError("Cached training/metadata/availability input mismatch: " + name)
        for name, value in cached_manifest["source_hashes"].items():
            recorded_source = ROOT/config["cache_run"]/"source-snapshot"/name
            if sha256(recorded_source if recorded_source.is_file() else ROOT/name) != value:
                raise ValueError("Cached source changed: " + name)
    target.mkdir(parents=True)
    started = time.perf_counter()
    env = environment()
    try:
        env["packages"]["implicit"] = importlib.metadata.version("implicit")
    except importlib.metadata.PackageNotFoundError:
        env["packages"]["implicit"] = None
    manifest = {"status": "running", "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "config_sha256": sha256(config_path), "source_hashes": sources(), "environment": env,
                "cache_manifest_sha256": sha256(ROOT / config["cache_run"] / "manifest.json"),
                "reuse_cached_fits": config.get("reuse_cached_fits", True),
                "cached_states_sha256": {b["name"]: sha256(ROOT / config["cache_run"] / b["name"] / "fitted-components.pkl") for b in config["slices"]} if config.get("reuse_cached_fits", True) else {},
                "extra_input_hashes": {**{str(ROOT / config["rate_features_root"] / name): value for name,value in config.get("rate_features_hashes", {}).items()},
                                       **{str(ROOT / config["engagement"]["root"] / name): value for name,value in config.get("engagement", {}).get("hashes", {}).items()},
                                       **{str(ROOT/config["inductive_mf"]["root"]/name):value for name,value in config.get("inductive_mf",{}).get("hashes",{}).items()}},
                "prepared": str(prepared), "template": str(Path(args.template).resolve()), "expected_users": args.expected_users,
                "preparation_manifest_sha256": sha256(prepared / "preparation-manifest.json"),
                "input_hashes": {str(p.relative_to(prepared)): sha256(p) for p in sorted(prepared.rglob("*")) if p.is_file()},
                "synthetic_fixture": args.synthetic_fixture, "external_uploads": False, "kaggle_score_objective": False}
    dump_json(target / "manifest.json", manifest)
    (target / "frozen-config.json").write_bytes(config_path.read_bytes())
    try:
        summaries = {}
        for block in config["slices"]:
            print(json.dumps({"phase": "slice_fit", "slice": block["name"]}), flush=True)
            summaries[block["name"]] = evaluate_slice(prepared, target / block["name"], block, config, started)
        selection = select_strategy(summaries, config)
        dump_json(target / "selection.json", selection)
        paired_bootstrap(target, selection, config)
        final = refit_final(prepared, target / "final", config, selection, args.template, args.expected_users)
        if sources() != manifest["source_hashes"]:
            raise RuntimeError("V3 source/config/tests changed during actual run")
        for name, value in manifest["input_hashes"].items():
            if sha256(prepared / name) != value:
                raise RuntimeError("Prepared input changed: " + name)
        for name, value in manifest["extra_input_hashes"].items():
            if sha256(Path(name)) != value:
                raise RuntimeError("Extra frozen exposure input changed: " + name)
        summary = {"status": "complete_synthetic_fixture" if args.synthetic_fixture else "complete_local_target040_round",
                   "selected_method": selection["selected_method"], "recommended_replacement_of_previous_best": selection["recommended_replacement_of_previous_best"],
                   "aggregate_criterion": selection["aggregate_criterion"], "per_slice_criterion": selection["per_slice_criterion"],
                   "raw_best_method": selection["raw_best_method"], "final_submission_sha256": final["sha256"],
                   "completed_trials": sum(len(result["methods"]) for result in summaries.values()),
                   "failed_trials": sum(len(result["failures"]) for result in summaries.values()),
                   "external_uploads": False, "kaggle_score": None, "hidden_evaluator_equivalence_claimed": False}
        dump_json(target / "summary.json", summary)
        manifest["status"] = summary["status"]
    except Exception as error:
        manifest["status"] = "failed"
        dump_json(target / "failure.json", {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
        raise
    finally:
        manifest["wall_seconds"] = time.perf_counter() - started
        manifest["peak_rss_bytes"] = peak_rss()
        manifest["source_hashes_at_end"] = sources()
        manifest["source_changed_during_run"] = manifest["source_hashes_at_end"] != manifest["source_hashes"]
        manifest["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        dump_json(target / "manifest.json", manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config/target040-round01.json")
    parser.add_argument("--template", type=Path, default=ROOT / "data/raw/extracted/submission.csv")
    parser.add_argument("--expected-users", type=int, default=1260)
    parser.add_argument("--synthetic-fixture", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
