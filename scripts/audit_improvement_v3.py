#!/usr/bin/env python3
"""Independent actual-export checks: targets, AP, fusion, retrieval and preservation.

This script does not import production metrics, models or selection functions.
It consumes the saved recommendation/native/contribution files and source inputs.
"""
import argparse
from collections import Counter, defaultdict
import csv
from fractions import Fraction
import gzip
import hashlib
import json
import math
from pathlib import Path

import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads(Path(path).read_text())


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def rows(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="", encoding="utf-8-sig") as stream:
        yield from csv.DictReader(stream)


def exact_ap(predictions, relevant):
    first = {}
    for rank, item in enumerate(predictions[:5], 1):
        if item in relevant:
            first.setdefault(item, rank)
    hits = sorted(first.values())
    return sum((Fraction(i, rank) for i, rank in enumerate(hits, 1)), Fraction(0)) / min(5, len(relevant)) if relevant else Fraction(0)


def rank_map(records):
    return {item: (i, score) for i, (item, score) in enumerate(records, 1)}


def history_rank(native, user, constant):
    scores = defaultdict(float)
    for component in ["history_seconds", "history_watches", "history_recent"]:
        for rank, (item, _) in enumerate(native[user][component], 1):
            scores[item] += (1 / 3) / (constant + rank)
    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))


def component_spec(spec, native, history, user):
    if spec["kind"] == "history_fusion":
        return {name: (rank_map(native[user][name]), 1 / 3) for name in ["history_seconds", "history_watches", "history_recent"]}
    if spec["kind"] not in {"hybrid", "hybrid_als"}:
        return None
    a, b, c = spec["weights"]
    result = {"ppr_v2": (rank_map(native[user]["ppr_v2"]), a),
              "personal_history_fusion": (rank_map(history[user]), b),
              "channel_time": (rank_map(native[user]["channel_time"]), c)}
    if spec["kind"] == "hybrid_als":
        result["ppr_v2"] = (result["ppr_v2"][0], a / 2)
        result["als32"] = (rank_map(native[user]["als32"]), a / 2)
    return result


def audit_slice(run, prepared, block, config, template):
    source, destination = prepared / block["name"], run / block["name"]
    summary = load(destination / "summary.json")
    train = pd.read_parquet(source / "train-interactions.parquet")
    heldout = pd.read_parquet(source / "valid-interactions.parquet")
    candidates = set(load(source / "candidates.json")["valid"])
    quantities = defaultdict(lambda: defaultdict(lambda: [0., 0]))
    for row in heldout.itertuples(index=False):
        assert type(row.user_id) is str and row.item_id > 0
        if row.item_id in candidates:
            value = quantities[row.user_id][int(row.item_id)]
            value[0] += float(row.seconds)
            value[1] += int(row.watches)
    cohort = [user for user in template if quantities[user]]
    assert cohort == load(destination / "cohort-and-targets.json")["user_ids"]
    targets = {kind: {} for kind in ["primary", "secondary", "all_observed"]}
    for user in cohort:
        records = quantities[user]
        targets["primary"][user] = set(sorted(records, key=lambda item: (-records[item][0], item))[:5])
        targets["secondary"][user] = set(sorted(records, key=lambda item: (-records[item][1], item))[:5])
        targets["all_observed"][user] = set(records)
    predictions, details = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list))
    metadata = {int(item): record for item, record in load(source / "metadata.json")["valid"].items()}
    for row in rows(destination / "recommendations.csv"):
        method, user, item = row["method"], row["user_id"], int(row["item_id"])
        assert user in template and item in candidates and math.isfinite(float(row["score"]))
        assert int(row["rank"]) == len(predictions[method][user]) + 1
        assert row["title"] == metadata.get(item, {}).get("tv_show_title", "")
        assert row["category"] == metadata.get(item, {}).get("tv_show_category", "")
        predictions[method][user].append(item)
        details[method][user].append(row)
    assert set(predictions) == set(summary["methods"])
    for method, users in predictions.items():
        assert [user for user in users if user in targets["primary"]] == cohort
        assert all(len(items) == len(set(items)) == 5 for items in users.values())
    exact_maps, count = {}, 0
    per_user = {}
    for kind in targets:
        values = defaultdict(list)
        per_user[kind] = defaultdict(dict)
        for row in rows(destination / ("per-user-ap-" + kind + ".csv")):
            user, method = row["user_id"], row["method"]
            assert user in targets[kind] and method in predictions
            assert set(json.loads(row["relevant_ids"])) == targets[kind][user]
            assert json.loads(row["predictions"]) == predictions[method][user]
            exact = exact_ap(predictions[method][user], targets[kind][user])
            assert abs(float(exact) - float(row["ap_at_5"])) < 1e-12
            assert user not in per_user[kind][method]
            per_user[kind][method][user] = exact
            values[method].append(exact)
            count += 1
        assert set(values) == set(summary["methods"])
        exact_maps[kind] = {}
        for method, scores in values.items():
            assert list(per_user[kind][method]) == cohort
            mean = sum(scores, Fraction(0)) / len(cohort)
            saved = summary["methods"][method]["metrics"][kind]
            assert abs(float(mean) - saved["map_at_5"]) < 1e-12
            assert str(mean) == saved["exact_macro_mean"]
            exact_maps[kind][method] = mean
    native = defaultdict(lambda: defaultdict(list))
    for row in rows(destination / "native-scores.csv.gz"):
        name, user, item, score = row["component"], row["user_id"], int(row["item_id"]), float(row["score"])
        assert row["popularity_padding"] == "False" and score > 0 and math.isfinite(score) and item in candidates
        assert int(row["rank"]) == len(native[user][name]) + 1
        native[user][name].append((item, score))
    scored_users = list(next(iter(predictions.values())))
    component_names = set(load(destination / "native-score-diagnostics.json"))
    for user in scored_users:
        for name in component_names:
            native[user].setdefault(name, [])
            records = native[user][name]
            assert len(records) <= 50 and len({i for i, _ in records}) == len(records)
            assert records == sorted(records, key=lambda pair: (-pair[1], pair[0]))
    history = {user: history_rank(native, user, config["fusion"]["rrf_constant"]) for user in scored_users}
    specs = {spec["name"]: spec for spec in config["methods"]}
    fusion_maps, expected_contributions = {}, 0
    for method in predictions:
        if specs[method]["kind"] in {"history_fusion", "hybrid", "hybrid_als"}:
            for user in scored_users:
                maps = component_spec(specs[method], native, history, user)
                fusion_maps[method, user] = maps
                expected_contributions += sum(len(ranks) for ranks, weight in maps.values())
    actual_contributions, group, seen = 0, None, set()
    fused_scores = defaultdict(lambda: defaultdict(float))
    for row in rows(destination / "fusion-contributions.csv.gz"):
        key = row["method"], row["user_id"]
        if key != group:
            group, seen = key, set()
        item, component = int(row["item_id"]), row["component"]
        assert (item, component) not in seen
        seen.add((item, component))
        ranks, weight = fusion_maps[key][component]
        rank, score = ranks[item]
        assert int(row["rank"]) == rank and abs(float(row["native_score"]) - score) < 1e-12
        assert abs(float(row["weight"]) - weight) < 1e-15
        contribution = weight / (config["fusion"]["rrf_constant"] + rank)
        assert abs(float(row["contribution"]) - contribution) < 1e-15
        fused_scores[key][item] += contribution
        actual_contributions += 1
    assert actual_contributions == expected_contributions
    popularity = train.groupby("item_id").user_id.nunique().to_dict()
    fallback_order = sorted(candidates, key=lambda item: (-popularity.get(item, 0), item))
    for method, users in predictions.items():
        for user, ids in users.items():
            if (method, user) in fusion_maps:
                scores = fused_scores[method, user]
            else:
                scores = dict(native[user][method])
            ordered = sorted(((item, value) for item, value in scores.items() if value > 0), key=lambda pair: (-pair[1], pair[0]))[:5]
            chosen = {item for item, _ in ordered}
            for item in fallback_order:
                if len(ordered) == 5:
                    break
                if item not in chosen:
                    ordered.append((item, float(popularity.get(item, 0))))
                    chosen.add(item)
            assert ids == [item for item, _ in ordered], (block["name"], method, user)
            assert all(abs(float(row["score"]) - expected[1]) < 1e-12 for row, expected in zip(details[method][user], ordered))
    union_recalls = {method: {kind: [0., 0., 0, 0, 0] for kind in ["primary", "secondary"]} for method in predictions}
    seen_union_users = defaultdict(list)
    for row in rows(destination / "candidate-unions.csv.gz"):
        method, user = row["method"], row["user_id"]
        assert user in cohort
        seen_union_users[method].append(user)
        kind = specs[method]["kind"]
        if kind in {"hybrid", "hybrid_als"}:
            names = ["ppr_v2", "history_seconds", "history_watches", "history_recent", "channel_time"]
            if kind == "hybrid_als":
                names.append("als32")
            expected = {item for name in names for item, _ in native[user][name]}
        elif kind == "history_fusion":
            expected = {item for item, _ in history[user]}
        else:
            expected = {item for item, _ in native[user][method]}
        actual = set(json.loads(row["native_ids"]))
        assert actual == expected and actual <= candidates and len(actual) <= 300
        fallback = {int(r["item_id"]) for r in details[method][user] if "training_popularity" in r["reason"]}
        assert fallback == set(json.loads(row["fallback_ids"]))
        for label in ["primary", "secondary"]:
            relevant = targets[label][user]
            a, b = len(relevant & actual), len(relevant & (actual | fallback))
            assert abs(float(row[label + "_native_recall"]) - a / len(relevant)) < 1e-12
            assert abs(float(row[label + "_with_fallback_recall"]) - b / len(relevant)) < 1e-12
            assert float(per_user[label][method][user]) <= b / len(relevant) + 1e-12
            values = union_recalls[method][label]
            values[0] += a / len(relevant)
            values[1] += b / len(relevant)
            values[2] += len(relevant)
            values[3] += a
            values[4] += b
    for method in predictions:
        assert seen_union_users[method] == cohort
        for kind, values in union_recalls[method].items():
            saved = summary["methods"][method]["candidate_recall"][kind]
            assert abs(saved["macro_native_union_recall"] - values[0] / len(cohort)) < 1e-12
            assert abs(saved["macro_union_plus_actual_fallback_recall"] - values[1] / len(cohort)) < 1e-12
            assert [saved["target_positions"], saved["native_union_hits"], saved["union_plus_fallback_hits"]] == values[2:]
    lengths = train.groupby("user_id").item_id.nunique().to_dict()
    profile = load(prepared / "context" / block["name"] / "profile.json")
    context_totals = sparse.load_npz(prepared / "context" / block["name"] / "counts.npz").sum(axis=1).A.ravel()
    raw_users = {user for user, total in zip(profile["users"], context_totals) if total > 0}
    segments = defaultdict(list)
    for user in cohort:
        n = lengths.get(user, 0)
        segment = "channel_only" if n == 0 and user in raw_users else "no_history" if n == 0 else "short" if n <= 20 else "medium" if n < 100 else "long"
        segments[segment].append(user)
    frozen_segments = load(destination / "segments.json")["users"]
    assert all(segments[name] == users for name, users in frozen_segments.items())
    for method in predictions:
        for kind in targets:
            for segment, users in frozen_segments.items():
                saved = summary["methods"][method]["metrics"][kind]["segments"][segment]
                assert saved["users"] == len(users) and saved["small_denominator"] == (len(users) < 30)
                if users:
                    exact = sum((per_user[kind][method][u] for u in users), Fraction(0)) / len(users)
                    assert abs(float(exact) - saved["map_at_5"]) < 1e-12
    global_items = set(int(item) for item in train.item_id)
    train_pairs = set(zip(train.user_id, train.item_id))
    for user in scored_users:
        for component in ["ppr_v2", "original_ppr_0.5", "ppr_cosine", "ppr_lift", "als32", "als64", "history_seconds", "history_watches", "history_recent"]:
            if component in native[user]:
                assert all(item in global_items for item, _ in native[user][component])
    counts = {}
    for kind in ["primary", "secondary"]:
        counts[kind] = {"positions": sum(len(v) for v in targets[kind].values()),
                        "seen_by_user": sum((u, i) in train_pairs for u, values in targets[kind].items() for i in values),
                        "absent_from_train": sum(i not in global_items for values in targets[kind].values() for i in values)}
        for label, predicate in [
                ("seen_by_same_user", lambda user, item: (user, item) in train_pairs),
                ("absent_from_all_train", lambda user, item: item not in global_items),
                ("known_but_not_seen_by_user", lambda user, item: item in global_items and (user, item) not in train_pairs)]:
            for method in predictions:
                subsets = {user: {item for item in relevant if predicate(user, item)} for user, relevant in targets[kind].items()}
                positions = sum(map(len, subsets.values()))
                hits = sum(len(relevant & set(predictions[method][user])) for user, relevant in subsets.items())
                n_users = sum(bool(relevant) for relevant in subsets.values())
                saved = summary["methods"][method]["metrics"][kind]["target_subsets"][label]
                assert [saved["target_positions"], saved["hits_at_5"], saved["users_with_subset"]] == [positions, hits, n_users]
                assert saved["small_denominator"] == (n_users < 30)
                assert saved["micro_recall_at_5"] is None if not positions else abs(saved["micro_recall_at_5"] - hits / positions) < 1e-12
    return {"cohort_size": len(cohort), "candidate_count": len(candidates), "AP_rows_checked": count,
            "fusion_contributions_checked": actual_contributions, "target_positions": counts,
            "exact_maps": {kind: {method: str(value) for method, value in values.items()} for kind, values in exact_maps.items()},
            "segments": {name: len(users) for name, users in frozen_segments.items()}, "all_native_rank_fusion_and_fallback_predictions_reproduced": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--prechange", type=Path, default=ROOT / "evidence/improvement-v3-01/prechange-manifest.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--metadata-change-audit", type=Path,
                        help="Explicit recorded .DS_Store-only hash exceptions; never permits data/model/source changes")
    args = parser.parse_args()
    run = args.run_dir.resolve()
    destination = args.output or run / "independent-audit.json"
    if destination.exists():
        raise FileExistsError("Independent audit must be saved under a new name")
    manifest, config, selection = load(run / "manifest.json"), load(run / "frozen-config.json"), load(run / "selection.json")
    assert manifest["status"] == "complete_local_v3_evaluation" and not manifest["source_changed_during_run"]
    assert digest(run / "frozen-config.json") == manifest["config_sha256"]
    prepared = Path(manifest["prepared"])
    for name, value in manifest["input_hashes"].items():
        assert digest(prepared / name) == value
    template = [row["user_id"] for row in rows(manifest["template"])]
    results = {block["name"]: audit_slice(run, prepared, block, config, template) for block in config["slices"]}
    complete = set.intersection(*(set(result["exact_maps"]["primary"]) for result in results.values()))
    criteria, per_slice = {}, {}
    for name in complete:
        per_slice[name] = {block: (Fraction(result["exact_maps"]["primary"][name]) + Fraction(result["exact_maps"]["secondary"][name])) / 2
                           for block, result in results.items()}
        criteria[name] = sum(per_slice[name].values(), Fraction(0)) / len(results)
    ranked = sorted(criteria, key=lambda name: (-criteria[name], config["selection"]["tie_order"].index(name)))
    best = ranked[0]
    wins = sum(per_slice[best][block] > per_slice["ppr_v2"][block] for block in results)
    replacement = best != "ppr_v2" and criteria[best] > criteria["ppr_v2"] and wins >= 2
    selected = best if replacement else "ppr_v2"
    assert selected == selection["selected_method"] and best == selection["raw_best_method"]
    assert replacement == selection["recommended_replacement_of_v2"]
    assert {name: str(value) for name, value in criteria.items()} == selection["exact_aggregate_criterion"]
    first = config["slices"][0]["name"]
    legacy_map = {"popularity": "original_popularity", "content": "original_content", "original_ppr_0.5": "original_ppr_0.5",
                  "ppr_v2": "improved_ppr_0.25_seconds_sqrt", "history_seconds": "history_seconds", "history_watches": "history_watches"}
    old = {(r["method"], r["user_id"], r["rank"]): (int(r["item_id"]), float(r["score"]))
           for r in rows(ROOT / "experiments/improvement-v2-01/development/recommendations.csv")}
    checked = 0
    for row in rows(run / first / "recommendations.csv"):
        if row["in_cohort"] == "True" and row["method"] in legacy_map:
            expected = old[legacy_map[row["method"]], row["user_id"], row["rank"]]
            assert expected[0] == int(row["item_id"]) and abs(expected[1] - float(row["score"])) < 1e-12
            checked += 1
    assert checked == 6 * results[first]["cohort_size"] * 5
    final = list(rows(run / "final/submission.csv"))
    candidates = set(load(prepared / "final/candidates.json")["valid"])
    assert [row["user_id"] for row in final] == template and len(template) == manifest["expected_users"]
    for row in final:
        text = row["tv_show_id"]
        tokens = text.split(" ")
        assert len(tokens) == 5 and all(token.isascii() and token.isdecimal() for token in tokens)
        ids = list(map(int, tokens))
        assert len(set(ids)) == 5 and set(ids) <= candidates and min(ids) > 0
        assert text == " ".join(map(str, ids))
    validation = load(run / "final/submission-validation.json")
    assert digest(run / "final/submission.csv") == validation["sha256"]
    assert (run / "selection.json").stat().st_mtime_ns <= (run / "final/model.pkl").stat().st_mtime_ns
    protected = load(args.prechange)
    exceptions = {}
    if args.metadata_change_audit:
        recorded = load(args.metadata_change_audit)
        for change in recorded["changed"]:
            assert Path(change["path"]).name == ".DS_Store"
            assert change["expected"] == protected["files"][change["path"]]
            exceptions[change["path"]] = change
    unchanged = 0
    for name, record in protected["files"].items():
        path = Path(name) if Path(name).is_absolute() else ROOT / name
        actual = {"bytes": path.stat().st_size, "sha256": digest(path)}
        if name in exceptions:
            assert actual == exceptions[name]["actual"], name
        else:
            assert actual == record, name
            unchanged += 1
    result = {"passed": True, "performed_by": "Codex independent actual-export audit, not author personal verification",
              "script_sha256": digest(__file__), "slices": results, "legacy_prediction_slots_identical": checked,
              "selected_method": selected, "selection_reproduced_exactly": True, "strict_slice_wins": wins,
              "protected_original_files_checked": len(protected["files"]),
              "protected_original_files_unchanged": unchanged,
              "protected_metadata_changes": list(exceptions.values()), "final_rows": len(final),
              "final_csv_sha256": validation["sha256"], "kaggle_uploaded": False, "kaggle_score": None}
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"passed": True, "selected": selected, "AP_rows": sum(v["AP_rows_checked"] for v in results.values()),
                      "fusion_rows": sum(v["fusion_contributions_checked"] for v in results.values()),
                      "legacy_slots": checked, "protected_files_unchanged": unchanged,
                      "explicit_metadata_exceptions": len(exceptions), "output": str(destination)}))


if __name__ == "__main__":
    main()
