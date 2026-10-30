#!/usr/bin/env python3
"""Independently check actual v2 exports and compare them with the original run."""
import argparse
from collections import defaultdict
import csv
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def load(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def csv_rows(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def ap_exact(predictions, relevant):
    seen, hits, total = set(), 0, Fraction(0)
    for rank, item in enumerate(predictions[:5], 1):
        if item in relevant and item not in seen:
            hits += 1
            total += Fraction(hits, rank)
        seen.add(item)
    return total / min(5, len(relevant)) if relevant else Fraction(0)


def audit_block(run, block, prepared):
    directory = run / block
    summary = load(directory / "summary.json")
    heldout = pd.read_parquet(prepared / "valid-interactions.parquet")
    candidates = set(load(prepared / "candidates.json")["valid"])
    template = csv_rows(ROOT / "data/raw/extracted/submission.csv")
    template_ids = [row["user_id"] for row in template]
    quantities = defaultdict(lambda: defaultdict(lambda: [0., 0.]))
    for row in heldout.itertuples(index=False):
        if int(row.item_id) in candidates:
            values = quantities[row.user_id][int(row.item_id)]
            values[0] += float(row.seconds)
            values[1] += float(row.watches)
    cohort = [user for user in template_ids if quantities[user]]
    expected = {"primary": {}, "secondary": {}, "all_observed": {}}
    for user in cohort:
        values = quantities[user]
        expected["primary"][user] = set(sorted(values, key=lambda item: (-values[item][0], item))[:5])
        expected["secondary"][user] = set(sorted(values, key=lambda item: (-values[item][1], item))[:5])
        expected["all_observed"][user] = set(values)
    methods = set(summary["methods"])
    by_kind = {}
    for kind in expected:
        per_method = defaultdict(list)
        predictions_by_method = defaultdict(dict)
        for row in csv_rows(directory / ("per-user-ap-" + kind + ".csv")):
            user, method = row["user_id"], row["method"]
            assert user in expected[kind] and method in methods
            relevant = set(json.loads(row["relevant_ids"]))
            predictions = json.loads(row["predictions"])
            assert relevant == expected[kind][user]
            assert len(predictions) == len(set(predictions)) == 5
            assert set(predictions) <= candidates
            value = ap_exact(predictions, relevant)
            assert abs(float(value) - float(row["ap_at_5"])) < 1e-12
            assert user not in predictions_by_method[method]
            predictions_by_method[method][user] = predictions
            per_method[method].append(value)
        assert set(per_method) == methods
        result = {}
        for method, values in per_method.items():
            assert list(predictions_by_method[method]) == cohort
            mean = sum(values, Fraction(0)) / len(cohort)
            assert abs(float(mean) - summary["methods"][method]["metrics"][kind]["map_at_5"]) < 1e-12
            result[method] = {"users": len(values), "exact_macro_mean": str(mean), "map_at_5": float(mean)}
        by_kind[kind] = result
    return {"cohort_size": len(cohort), "candidate_count": len(candidates), "independent_targets_and_AP": by_kind}


def audit_selected_state(run, original, selected, spec):
    with (run / "final/model.pkl").open("rb") as stream:
        model = pickle.load(stream)
    with (original / "final/model.pkl").open("rb") as stream:
        original_model = pickle.load(stream)
    result = {"kind": spec["kind"]}
    if spec["kind"] != "improved_ppr":
        result["stronger_seed_check_applicable"] = False
        return result
    assert model.data.users == original_model.data.users
    assert np.array_equal(model.data.items, original_model.data.items)
    assert (model.transition != original_model.transition).nnz == 0
    train = pd.read_parquet(original / "prepared/full-interactions.parquet")
    signal = spec["seed_signal"]
    if spec.get("recency"):
        train = pd.read_parquet(ROOT / "experiments/improvement-v2-data/recency/full.parquet")
        signal = "recent_seconds"
    grouped = train.groupby(["user_id", "item_id"])[signal].sum()
    # Check every stored restart against an independently aggregated quantity.
    maximum_seed_error = 0.
    for user, records in grouped.groupby(level=0):
        values = {int(item): float(value) ** spec["seed_power"] for (_, item), value in records.items()}
        total = sum(values.values())
        observed = model.restart_rows.getrow(model.data.user_index[user])
        expected = np.zeros(len(model.data.items), dtype=np.float64)
        for item, value in values.items():
            expected[model.data.item_index[item]] = value / total
        maximum_seed_error = max(maximum_seed_error, float(np.max(np.abs(observed.toarray().ravel() - expected))))
    assert maximum_seed_error < 1e-12
    sampled = []
    # Independent high-precision fixed-point iteration, without calling _walk.
    for user in ["13571199046623737884", model.data.users[0], model.data.users[-1]]:
        if user not in model.data.user_index:
            continue
        seed = model.restart_rows.getrow(model.data.user_index[user]).toarray().ravel()
        scores = seed.copy()
        for iteration in range(10000):
            next_scores = ((1-model.alpha) + model.alpha * scores[model.dangling].sum()) * seed + model.alpha * model.transition.T.dot(scores)
            delta = np.abs(next_scores - scores).sum()
            scores = next_scores
            if delta < 1e-13:
                break
        actual = model._walk(user)
        error = float(np.abs(actual-scores).sum())
        assert abs(scores.sum()-1) < 1e-12 and error < 1e-7
        sampled.append({"user_id": user, "independent_iterations": iteration+1, "l1_error_vs_checkpoint": error, "probability_mass": float(scores.sum())})
    result.update({"graph_exactly_equals_v1_full_refit": True, "all_users_seed_max_absolute_error": maximum_seed_error, "independent_stationary_checks": sampled})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "experiments/improvement-v2-01")
    parser.add_argument("--output", type=Path, default=ROOT / "evidence/improvement-v2-independent-audit.json")
    args = parser.parse_args()
    run = args.run_dir.resolve()
    original = ROOT / "experiments/competition-20261030-03"
    manifest, summary = load(run / "manifest.json"), load(run / "summary.json")
    assert manifest["status"] == "complete_local_v2_proxy_evaluation"
    assert not manifest["source_changed_during_run"]
    config, selection = load(run / "frozen-config.json"), load(run / "selection.json")
    development = audit_block(run, "development", ROOT / "experiments/improvement-v2-data/development")
    confirmation = audit_block(run, "confirmation", original / "prepared")
    eligible = [spec for spec in config["methods"] if spec["selection_eligible"]]
    selected = max(eligible, key=lambda spec: development["independent_targets_and_AP"]["primary"][spec["name"]]["map_at_5"])
    assert selected == selection["selected_specification"]
    assert selected["name"] == summary["selected_method"]
    v1 = {(row["method"], row["user_id"], row["rank"]): row["item_id"] for row in csv_rows(original / "validation-recommendations.csv")}
    checked = 0
    names = {"original_popularity": "popularity", "original_content": "content", "original_ppr_0.5": "ppr_0.5"}
    for row in csv_rows(run / "confirmation/recommendations.csv"):
        if row["method"] in names:
            key = (names[row["method"]], row["user_id"], row["rank"])
            assert v1[key] == row["item_id"]
            checked += 1
    old_csv, new_csv = csv_rows(original / "final/submission.csv"), csv_rows(run / "final/submission.csv")
    assert [row["user_id"] for row in old_csv] == [row["user_id"] for row in new_csv]
    changed = sum(old["tv_show_id"] != new["tv_show_id"] for old, new in zip(old_csv, new_csv))
    assert digest(run / "final/submission.csv") == summary["final_submission_sha256"]
    result = {"passed": True, "performed_by": "Codex independent automated audit, not student personal verification", "audit_script_sha256": digest(__file__), "development": development, "confirmation": confirmation, "selection_reproduced_from_development_only": True, "original_three_baselines_identical_confirmation_slots": checked, "selected_state": audit_selected_state(run, original, selected["name"], selected), "final_comparison": {"users": len(new_csv), "changed_rows_vs_v1": changed, "original_sha256": digest(original / "final/submission.csv"), "new_sha256": digest(run / "final/submission.csv")}, "hidden_evaluator_identity_established": False, "v2_kaggle_score_obtained": False}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"passed": True, "output": str(args.output), "selected": selected["name"], "changed_final_rows": changed}))


if __name__ == "__main__":
    main()
