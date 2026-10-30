"""MAP@5 and an independent, exact audit of exported validation results.

Empty relevant sets receive AP zero.  ``evaluate`` uses every member of the
explicit cohort, including empty-target users, and never infers a warm cohort.
The exported-file audit deliberately does not call the production AP function.
"""

from __future__ import annotations

import csv
from fractions import Fraction
import json
import math
from pathlib import Path
from typing import Hashable, Iterable, Mapping, Sequence


def ap_at_k(
    relevant: Iterable[Hashable], predictions: Iterable[Hashable], k: int = 5
) -> float:
    """Return AP with unique-label denominator and distinct-hit rank credit.

    A repeated prediction keeps its rank position but cannot earn another hit.
    The denominator is ``min(k, number of unique relevant labels)``.  Empty
    labels and nonpositive ``k`` have AP zero.
    """
    labels = set(relevant)
    if k <= 0 or not labels:
        return 0.0
    seen: set[Hashable] = set()
    hits = 0
    precision_sum = 0.0
    for rank, item in enumerate(predictions, start=1):
        if rank > k:
            break
        if item in labels and item not in seen:
            hits += 1
            precision_sum += hits / rank
        seen.add(item)
    return precision_sum / min(k, len(labels))


def evaluate(
    recommendations: Mapping[str, Sequence[int]],
    relevant: Mapping[str, Iterable[int]],
    cohort: Sequence[str],
) -> dict:
    """Calculate per-user AP@5 and a macro mean over one explicit cohort.

    The caller must provide both recommendations and targets for each cohort
    user.  Empty targets must therefore be represented explicitly, rather than
    silently arising from an omitted key.  Additional mapping keys are ignored.
    """
    users = list(cohort)
    if not users:
        raise ValueError("The fixed evaluation cohort must not be empty")
    if len(set(users)) != len(users):
        raise ValueError("The fixed evaluation cohort contains duplicate users")
    missing_predictions = [user for user in users if user not in recommendations]
    missing_targets = [user for user in users if user not in relevant]
    if missing_predictions or missing_targets:
        raise ValueError(
            "Every cohort user needs explicit predictions and targets; "
            f"missing_predictions={missing_predictions[:5]}, "
            f"missing_targets={missing_targets[:5]}"
        )
    targets = {user: set(relevant[user]) for user in users}
    per_user = {user: ap_at_k(targets[user], recommendations[user]) for user in users}
    return {
        "map_at_5": math.fsum(per_user.values()) / len(users),
        "cohort_size": len(users),
        "empty_target_users": sum(not targets[user] for user in users),
        "per_user_ap": per_user,
    }


def _summary_maps(payload: object) -> dict[str, float]:
    """Read method summaries without imposing a report-specific outer schema."""
    if not isinstance(payload, dict):
        raise ValueError("Metric summary must be a JSON object")
    if "methods" in payload:
        payload = payload["methods"]
        if not isinstance(payload, dict):
            raise ValueError("Metric summary methods must be a JSON object")
    result = {}
    for name, values in payload.items():
        if isinstance(values, dict) and "map_at_5" in values:
            result[str(name)] = float(values["map_at_5"])
        elif isinstance(values, (float, int)) and not isinstance(values, bool):
            result[str(name)] = float(values)
    if not result:
        raise ValueError("No method MAP@5 values found in metric summary")
    return result


def audit_exported_ap(
    per_user_csv_path: str | Path,
    summary_path: str | Path | None = None,
    *,
    expected_map: Mapping[str, float] | None = None,
    tolerance: float = 1e-12,
) -> dict:
    """Independently recompute exact AP@5 and macro MAP from the actual CSV.

    Required columns are ``user_id``, ``relevant_ids``, ``predictions`` and
    ``ap_at_5``; both list columns contain JSON arrays.  An optional ``method``
    column groups multiple models.  Each model must export the same ordered
    cohort.  Repeated labels and predictions are handled with exact Fraction
    arithmetic; saved floating AP and optional saved macro means are checked.

    ``expected_map`` is a mapping from method name to the corresponding summary
    MAP.  Alternatively, ``summary_path`` can name a JSON mapping containing
    either method-to-MAP values or method-to-objects with ``map_at_5`` fields,
    optionally wrapped in ``{"methods": ...}``.  A CSV without ``method`` uses
    the method name ``default``.
    """
    if tolerance < 0 or not math.isfinite(tolerance):
        raise ValueError("Audit tolerance must be nonnegative and finite")
    sums: dict[str, Fraction] = {}
    users: dict[str, list[str]] = {}
    user_sets: dict[str, set[str]] = {}
    empty_counts: dict[str, int] = {}
    max_ap_error = 0.0
    with Path(per_user_csv_path).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"user_id", "relevant_ids", "predictions", "ap_at_5"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Per-user export lacks columns {sorted(required)}")
        for line, row in enumerate(reader, start=2):
            method = row.get("method", "default")
            user = row["user_id"]
            if not method or not user or None in row or any(value is None for value in row.values()):
                raise ValueError(f"Malformed or empty user/method at CSV line {line}")
            try:
                labels_raw = json.loads(row["relevant_ids"])
                predicted = json.loads(row["predictions"])
                saved_ap = float(row["ap_at_5"])
            except (ValueError, TypeError) as error:
                raise ValueError(f"Invalid AP/JSON at CSV line {line}") from error
            if not isinstance(labels_raw, list) or not isinstance(predicted, list):
                raise ValueError(f"Labels and predictions must be JSON arrays at line {line}")
            if any(type(item) is not int or item <= 0 for item in labels_raw + predicted):
                raise ValueError(f"Programme IDs must be positive JSON integers at line {line}")
            if not math.isfinite(saved_ap):
                raise ValueError(f"Non-finite saved AP at line {line}")
            labels = set(labels_raw)
            # Independent definition: identify a relevant label's first rank,
            # then sum distinct-hit precisions over those ordered ranks.
            first_ranks = {}
            for rank, item in enumerate(predicted[:5], start=1):
                if item in labels and item not in first_ranks:
                    first_ranks[item] = rank
            hit_ranks = sorted(first_ranks.values())
            precision_sum = sum(
                (Fraction(hit_index, rank) for hit_index, rank in enumerate(hit_ranks, start=1)),
                Fraction(0),
            )
            exact_ap = precision_sum / min(5, len(labels)) if labels else Fraction(0)
            error = abs(float(exact_ap) - saved_ap)
            max_ap_error = max(max_ap_error, error)
            if error > tolerance:
                raise ValueError(
                    f"AP audit mismatch for method={method!r}, user={user!r}: "
                    f"saved={saved_ap}, recomputed={exact_ap}"
                )
            if user in user_sets.setdefault(method, set()):
                raise ValueError(f"Duplicate exported user {user!r} for {method!r}")
            user_sets[method].add(user)
            users.setdefault(method, []).append(user)
            sums[method] = sums.get(method, Fraction(0)) + exact_ap
            empty_counts[method] = empty_counts.get(method, 0) + int(not labels)
    if not sums:
        raise ValueError("Per-user export contains no observations")
    reference_users = next(iter(users.values()))
    if any(method_users != reference_users for method_users in users.values()):
        raise ValueError("Methods do not have the same ordered validation cohort")

    expected = dict(expected_map or {})
    if summary_path is not None:
        with Path(summary_path).open("r", encoding="utf-8") as stream:
            saved_summary = _summary_maps(json.load(stream))
        for method, value in saved_summary.items():
            if method in expected and abs(expected[method] - value) > tolerance:
                raise ValueError(f"Conflicting expected MAP values for {method!r}")
            expected[method] = value
    if expected and set(expected) != set(sums):
        raise ValueError("Summary methods differ from per-user export methods")

    methods = {}
    for method, exact_sum in sums.items():
        macro = exact_sum / len(users[method])
        if method in expected:
            value = float(expected[method])
            if not math.isfinite(value) or abs(float(macro) - value) > tolerance:
                raise ValueError(
                    f"Macro audit mismatch for {method!r}: "
                    f"saved={value}, recomputed={macro}"
                )
        methods[method] = {
            "map_at_5": float(macro),
            "exact_numerator": macro.numerator,
            "exact_denominator": macro.denominator,
            "cohort_size": len(users[method]),
            "empty_target_users": empty_counts[method],
        }
    return {"methods": methods, "max_per_user_ap_error": max_ap_error, "passed": True}
