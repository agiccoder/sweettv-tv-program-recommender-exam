#!/usr/bin/env python3
"""Check real ALS scores against independent float64 contraction after a run.

This is a diagnostic of the frozen models, not another quality trial or tuning.
It retains the numerical warnings seen with this NumPy/Apple BLAS environment.
Only locally generated, trusted model pickles may be supplied.
"""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import warnings

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    output = run / "independent-ALS-numerics.json"
    if output.exists():
        raise FileExistsError(output)
    config = json.loads((run / "frozen-config.json").read_text())
    result = {}
    for block in config["slices"]:
        with (run / block["name"] / "fitted-components.pkl").open("rb") as stream:
            family = pickle.load(stream)
        for name in ["als32", "als64"]:
            if name not in family.models:
                result[block["name"] + "/" + name] = {"unavailable": True}
                continue
            wrapper = family.models[name]
            users, items = wrapper.model.user_factors, wrapper.model.item_factors
            assert np.isfinite(users).all() and np.isfinite(items).all()
            bound = float(np.max(np.abs(users))) * float(np.max(np.abs(items))) * items.shape[1]
            assert bound < np.finfo(np.float32).max
            selected = list(dict.fromkeys(list(wrapper.data.users[:3]) + config["examples"] +
                                         [wrapper.data.users[int(np.argmax(np.linalg.norm(users, axis=1)))]]))
            maximum_absolute_error, comparisons, recorded = 0., 0, []
            for user in selected:
                index = wrapper.data.user_index.get(user)
                if index is None:
                    assert wrapper.native(user, family.pool) == {}
                    continue
                reference = np.einsum("ij,j->i", items.astype(np.float64), users[index].astype(np.float64), optimize=False)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    actual = wrapper.native(user, family.pool)
                recorded.extend(str(w.message) for w in caught)
                actual_values = np.array(list(actual.values()))
                expected = np.array([reference[wrapper.data.item_index[item]] for item in actual])
                assert np.isfinite(actual_values).all()
                assert np.allclose(actual_values, expected, rtol=1e-5, atol=2e-6)
                maximum_absolute_error = max(maximum_absolute_error, float(np.max(np.abs(actual_values - expected))))
                comparisons += len(actual)
            result[block["name"] + "/" + name] = {
                "user_item_factor_shapes": [list(users.shape), list(items.shape)],
                "all_factors_finite": True, "maximum_possible_absolute_dot_bound": bound,
                "reference": "Independent float64 numpy.einsum without BLAS matmul",
                "sample_user_ids": selected, "candidate_scores_compared": comparisons,
                "maximum_absolute_error": maximum_absolute_error,
                "tolerance": {"rtol": 1e-5, "atol": 2e-6},
                "warnings_during_check": sorted(set(recorded)), "passed": True}
    output.write_text(json.dumps({"passed": True, "models": result,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "Frozen real models; numerical diagnostic only, no refit or selection change"}, indent=2) + "\n")
    print(json.dumps({"passed": True, "models": len(result), "output": str(output)}))


if __name__ == "__main__":
    main()
