"""Versioned experiments with stronger personal restart signals.

This module leaves the original models and their results unchanged. These
variants are hypotheses for a frozen local validation protocol, not claims of
improved accuracy and not authorization for another competition submission.

The graph is exactly the original training-only programme co-view graph:
X[u,i] = log1p(seconds[u,i]) / sum_j log1p(seconds[u,j]); A = X.T @ X;
remove the diagonal, retain the deterministic top 100 neighbours, normalize
rows. Only the personal restart changes. For nonnegative row scale h_r,
z[u,i] = (sum_{rows r for u,i} h_r * signal_r) ** power, where signal is
seconds or accepted-airing watches. The restart is v[u] = z[u] / sum_i z[u,i].
The default power .5 compresses extremes less than log1p(seconds). Lower alpha
retains more personal restart mass; its utility must be established locally.

The original PPR walk, dangling redistribution, sparse capacity bounds,
convergence diagnostics, deterministic ranking and popularity fallback are
reused. Personal-history ranking by z is also provided as a diagnostic baseline;
it is not a second qualifying recommendation family. Repeats remain allowed.

Optional history_scale is a named interaction column, row-aligned iterable, or
mapping {(raw_user_id, item_id): nonnegative_scale}. Missing mapping keys use
one. Scale applies BEFORE per-user/item aggregation and exponentiation; it
never changes the graph or fallback. Recency weights must be constructed by the
caller from timestamps before that experiment's training cutoff. This module
does not infer timestamps, read validation labels, or fit at prediction time.
"""

from collections.abc import Mapping
import hashlib
import math
from typing import Any, Optional

import numpy as np
from scipy import sparse

from .models import _BaseModel, _TrainingData, _records, PPRModel


def _freeze(matrix):
    for array in (matrix.data, matrix.indices, matrix.indptr):
        array.flags.writeable = False
    return matrix


def _resolve_scales(records, history_scale):
    if history_scale is None:
        values = np.ones(len(records), dtype=np.float64)
        policy = "none; every row has scale one"
    elif isinstance(history_scale, str):
        try:
            values = np.asarray([row[history_scale] for row in records], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("history_scale column must exist with numeric values on every row") from error
        policy = "interaction column: " + history_scale
    elif isinstance(history_scale, Mapping):
        values = np.asarray([history_scale.get((row["user_id"], row["item_id"]), 1.) for row in records], dtype=np.float64)
        policy = "mapping (raw user_id,item_id) -> scale; missing keys use one"
    else:
        try:
            values = np.asarray(list(history_scale), dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError("history_scale must be a column name, key mapping or row-aligned numeric iterable") from error
        policy = "row-aligned numeric iterable"
    if values.ndim != 1 or len(values) != len(records):
        raise ValueError("history_scale must have exactly one value per interaction row")
    if np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError("history_scale values must be finite and nonnegative")
    return values, {
        "history_scale_policy": policy,
        "history_scale_min": float(values.min()) if len(values) else None,
        "history_scale_max": float(values.max()) if len(values) else None,
        "history_scale_zero_rows": int(np.count_nonzero(values == 0)),
        "history_scale_nonunit_rows": int(np.count_nonzero(values != 1)),
        "history_scale_row_values_sha256": hashlib.sha256(values.astype("<f8", copy=False).tobytes()).hexdigest(),
        "history_scale_boundary": "caller supplies training-only scales; no timestamps or held-out information are inferred here",
    }


def _prepare(interactions: Any, signal: str, power: float, history_scale=None):
    if signal not in {"seconds", "watches"}:
        raise ValueError("signal must be 'seconds' or 'watches'")
    if not isinstance(power, (int, float, np.number)) or not math.isfinite(float(power)) or power <= 0:
        raise ValueError("power must be positive and finite")
    # Materialize once: the graph and seeds consume identical immutable copies
    # of rows; subsequent changes to the caller's DataFrame cannot change state.
    records = [dict(row) for row in _records(interactions)]
    data = _TrainingData(records)
    scales, diagnostics = _resolve_scales(records, history_scale)
    aggregated = {}
    for row, scale in zip(records, scales):
        if float(row["seconds"]) <= 0:
            continue  # Match the original training graph's usable-history rule.
        key = (row["user_id"], int(row["item_id"]))
        value = float(row[signal]) * float(scale)
        total = aggregated.get(key, 0.) + value
        if not math.isfinite(total):
            raise ValueError("Scaled per-user/item history signal overflowed")
        aggregated[key] = total
    rows, columns, values = [], [], []
    for (user, item), total in sorted(aggregated.items()):
        if total <= 0:
            continue
        try:
            transformed = total ** float(power)
        except OverflowError as error:
            raise ValueError("Powered history signal overflowed") from error
        if not math.isfinite(transformed) or transformed <= 0:
            raise ValueError("Powered history signal must remain positive and finite")
        rows.append(data.user_index[user])
        columns.append(data.item_index[item])
        values.append(transformed)
    seeds = sparse.csr_matrix((values, (rows, columns)), shape=data.weights.shape, dtype=np.float64)
    sums = np.asarray(seeds.sum(axis=1)).ravel()
    if np.any(~np.isfinite(sums)):
        raise ValueError("Per-user seed normalization overflowed")
    restart = (sparse.diags(np.divide(1., sums, out=np.zeros_like(sums), where=sums > 0)) @ seeds).tocsr()
    restart.sort_indices()
    diagnostics.update({
        "seed_signal": signal, "seed_power": float(power),
        "seed_transform": "(sum of signal times row scale per user and item) ** power",
        "seed_normalization": "L1 per user for PPR; personal-history diagnostic ranks unnormalized powered values",
        "seed_nonzero_user_item_pairs": seeds.nnz,
        "zero_seed_training_users": int(np.count_nonzero(sums == 0)),
        "seed_state": "powered values and normalized restart cached as read-only CSR arrays",
    })
    return data, _freeze(seeds), _freeze(restart), diagnostics


class _FrozenSignalState:
    def __setstate__(self, state):
        self.__dict__.update(state)
        _freeze(self.seed_values)
        _freeze(self.restart_rows)


class ImprovedPPRModel(_FrozenSignalState, PPRModel):
    def __init__(self, data, seeds, restart, seed_diagnostics, alpha):
        # The original constructor builds the exact original graph. Its
        # normalized log-seconds restart is the only state replaced below.
        super().__init__(data, alpha=alpha, max_neighbors=100, tolerance=1e-8, max_iterations=300)
        self.seed_values = seeds
        self.restart_rows = restart
        self.diagnostics.update(seed_diagnostics)
        self.diagnostics.update({
            "method": "personalized_pagerank_power_restart",
            "graph_boundary": "original unscaled log-seconds graph; recency/seed transformations never alter edges",
            "qualifying_family": "personalized PageRank",
            "improvement_status": "experimental hypothesis; local validation required",
        })

    def recommend(self, user_id, candidates, k=5):
        pool = self._candidates(candidates, k)
        index = self.data.user_index.get(user_id)
        cold = index is None or self.restart_rows.getrow(index).nnz == 0
        scores = {}
        if not cold:
            values = self._walk(user_id)
            scores = {item: float(values[self.data.item_index[item]]) for item in pool if item in self.data.item_index}
        return self._rank(pool, scores, k, "personalized_pagerank_power_restart", cold)


class PersonalHistoryModel(_FrozenSignalState, _BaseModel):
    def __init__(self, data, seeds, restart, seed_diagnostics):
        super().__init__(data)
        self.seed_values = seeds
        self.restart_rows = restart
        self.diagnostics.update(seed_diagnostics)
        self.diagnostics.update({
            "method": "personal_history_diagnostic",
            "qualifying_family": None,
            "definition": "rank own accepted training history by powered aggregate signal; append common training-popularity fallback",
            "training_nonzero_score_variance": float(np.var(seeds.data)) if seeds.nnz else 0.,
            "improvement_status": "diagnostic baseline only; not a second qualifying recommendation family",
        })

    def recommend(self, user_id, candidates, k=5):
        pool = self._candidates(candidates, k)
        index = self.data.user_index.get(user_id)
        cold = index is None or self.seed_values.getrow(index).nnz == 0
        scores = {}
        if not cold:
            row = self.seed_values.getrow(index)
            scores = {int(self.data.items[item]): float(value) for item, value in zip(row.indices, row.data)}
        return self._rank(pool, scores, k, "personal_history_diagnostic", cold)


def fit_improved_ppr(interactions: Any, *, alpha=.15, seed_signal="seconds", seed_power=.5, history_scale=None) -> ImprovedPPRModel:
    """Fit original-graph PPR with an explicitly powered personal restart."""
    data, seeds, restart, diagnostics = _prepare(interactions, seed_signal, seed_power, history_scale)
    return ImprovedPPRModel(data, seeds, restart, diagnostics, alpha)


def fit_personal_history(interactions: Any, signal="seconds", power=1, *, history_scale=None) -> PersonalHistoryModel:
    """Own-history repeat baseline; this does not count as a qualifying family."""
    data, seeds, restart, diagnostics = _prepare(interactions, signal, power, history_scale)
    return PersonalHistoryModel(data, seeds, restart, diagnostics)


def fit_history(interactions: Any, *, seed_kind="seconds", power=.5, history_scale=None) -> PersonalHistoryModel:
    """Runner compatibility alias for the explicitly diagnostic history model."""
    return fit_personal_history(interactions, signal=seed_kind, power=power, history_scale=history_scale)
