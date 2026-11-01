"""Versioned native ranking, bounded graph experiments, implicit ALS and fixed RRF.

No popularity padding enters a fusion. All preferences are supplied by the
cutoff-specific preparation; schedule exposure contains availability fields only.
The original v1/v2 modules are deliberately unchanged.
"""
from __future__ import annotations

import math
from pathlib import Path
import pickle
import time

import numpy as np
from scipy import sparse

from .improved_models import _prepare
from .models import (ContentModel, PPRModel, PopularityModel, _BaseModel,
                     _pruned_transition)


def ordered_native(scores, limit=50):
    if any(not math.isfinite(float(value)) for value in scores.values()):
        raise ValueError("Non-finite native scores")
    return sorted(((int(item), float(value)) for item, value in scores.items() if value > 0),
                  key=lambda pair: (-pair[1], pair[0]))[:limit]


def native_scores(model, user, candidates):
    """Score known items without calling any recommend()/fallback padding."""
    if hasattr(model, "native"):
        return model.native(user, candidates)
    if isinstance(model, PopularityModel):
        return {item: model.data.popularity.get(item, 0) for item in candidates}
    if isinstance(model, PPRModel):
        if user not in model.data.user_index:
            return {}
        values = model._walk(user)
        return {item: float(values[model.data.item_index[item]]) for item in candidates if item in model.data.item_index}
    if isinstance(model, ContentModel):
        index = model.data.user_index.get(user)
        if index is None or model.profiles.getrow(index).nnz == 0:
            return {}
        known = [(item, model.metadata_index[item]) for item in candidates if item in model.metadata_index]
        if not known:
            return {}
        scores = (model.item_vectors[[i for _, i in known]] @ model.profiles.getrow(index).T).toarray().ravel()
        return {item: float(value) for (item, _), value in zip(known, scores)}
    raise TypeError(type(model))


def rrf(lists, weights, constant=60., allowed=None):
    """Fixed weighted reciprocal ranks, counting each ID once per component.

    Missing components contribute zero. Weights are never normalized per item.
    Native scores are exported as explanations; only ranks enter the sum.
    """
    if set(lists) != set(weights) or constant < 0:
        raise ValueError("RRF needs one explicit weight per component")
    if any(not math.isfinite(float(w)) or w < 0 for w in weights.values()):
        raise ValueError("RRF weights must be finite nonnegative")
    scores, contributions = {}, {}
    for component, records in lists.items():
        seen = set()
        for rank, (item, native_score) in enumerate(records, 1):
            if item in seen:
                raise ValueError("Duplicate native ID within a fusion component")
            seen.add(item)
            if allowed is not None and item not in allowed:
                continue
            value = weights[component] / (constant + rank)
            scores[item] = scores.get(item, 0.) + value
            contributions.setdefault(item, {})[component] = {
                "rank": rank, "weight": weights[component], "native_score": float(native_score), "contribution": value}
    return scores, contributions


class SignalModel(_BaseModel):
    def __init__(self, data, frame, signal):
        super().__init__(data)
        self.values, self.signal = {}, signal
        for user, item, value in frame[["user_id", "item_id", signal]].itertuples(index=False, name=None):
            if not isinstance(user, str) or not math.isfinite(float(value)) or value < 0:
                raise ValueError("Malformed personal history")
            history = self.values.setdefault(user, {})
            history[int(item)] = history.get(int(item), 0.) + float(value)
        self.diagnostics.update({"method": "history_" + signal, "signal": signal})

    def native(self, user, candidates):
        history = self.values.get(user, {})
        return {item: value for item, value in history.items() if item in candidates}


class ChannelModel(_BaseModel):
    def __init__(self, data, state, exposure, items):
        super().__init__(data)
        self.state, self.exposure, self.items = state, exposure, list(items)
        self.global_scores = np.asarray(exposure @ state.global_profile).ravel()
        self.diagnostics.update({"method": "channel_time", "training_raw_users": len(state.users),
                  "training_channels": len(state.channels), "training_context_nonzeros": state.counts.nnz,
                  "smoothing_seconds": state.smoothing_seconds, "future_content_features_used": False})

    def native(self, user, candidates):
        index = self.state.user_index.get(user)
        if index is None or self.state.totals[index] <= 0:
            return {}
        own = (self.exposure @ self.state.counts.getrow(index).T).toarray().ravel()
        scores = (own + self.state.smoothing_seconds * self.global_scores) / (self.state.totals[index] + self.state.smoothing_seconds)
        return {item: float(value) for item, value in zip(self.items, scores) if item in candidates}


def normalized_graph(data, restart, kind, config):
    if kind not in {"cosine", "lift"}:
        raise ValueError(kind)
    n = len(data.items)
    if n > config["max_training_items"]:
        raise ValueError("Graph capacity exceeded")
    row_totals = np.asarray(data.weights.sum(axis=1)).ravel()
    x = (sparse.diags(np.divide(1., row_totals, out=np.zeros_like(row_totals), where=row_totals > 0)) @ data.weights).tocsr()
    binary = data.weights.copy()
    binary.data[:] = 1.
    transposed, binary_t = x.T.tocsr(), binary.T.tocsr()
    diagonal = np.asarray(x.power(2).sum(axis=0)).ravel()
    mass = np.asarray(x.sum(axis=0)).ravel()
    rows, columns, values, before = [0], [], [], 0
    block_size = min(config["product_block_rows"], max(1, config["product_payload_budget_bytes"] // max(32 * n, 1)))
    for begin in range(0, n, block_size):
        covariance = (transposed[begin:begin + block_size] @ x).tocsr()
        support = (binary_t[begin:begin + block_size] @ binary).tocsr()
        for offset in range(covariance.shape[0]):
            i = begin + offset
            left, right = covariance.indptr[offset:offset + 2]
            js, numerators = covariance.indices[left:right], covariance.data[left:right]
            counts = support.getrow(offset).toarray().ravel()[js]
            keep = (js != i) & (counts >= config["minimum_support"]) & (numerators > 0)
            js, numerators, counts = js[keep], numerators[keep], counts[keep]
            if kind == "cosine":
                scores = numerators / np.sqrt(diagonal[i] * diagonal[js])
            else:
                smooth = config["lift_mass_smoothing"]
                scores = len(data.users) * numerators / ((mass[i] + smooth) * (mass[js] + smooth))
                scores *= counts / (counts + config["lift_support_shrinkage"])
            if np.any(~np.isfinite(scores)) or np.any(scores < 0):
                raise ValueError("Invalid normalized graph scores")
            before += len(js)
            order = np.lexsort((data.items[js], -scores))[:config["max_neighbors"]]
            columns.extend(js[order].tolist())
            values.extend(scores[order].tolist())
            rows.append(len(columns))
    adjacency = sparse.csr_matrix((values, columns, rows), shape=(n, n), dtype=np.float64)
    sums = np.asarray(adjacency.sum(axis=1)).ravel()
    transition = (sparse.diags(np.divide(1., sums, out=np.zeros_like(sums), where=sums > 0)) @ adjacency).tocsr()
    transition.sort_indices()
    diagnostics = {"normalization": kind, "minimum_support": config["minimum_support"],
                   "graph_weight": config[kind], "graph_product_block_rows": block_size,
                   "support_definition": config["support"], "graph_memory_bound": config["product_payload_budget_bytes"]}
    return transition, sums == 0, restart, before, diagnostics


class ALSModel(_BaseModel):
    def __init__(self, data, factors, config):
        from implicit.cpu.als import AlternatingLeastSquares
        import implicit
        super().__init__(data)
        self.confidence = data.weights.copy().astype(np.float32)
        self.confidence.data = (1 + config["confidence_alpha"] * np.log1p(np.expm1(data.weights.data) / config["seconds_scale"])).astype(np.float32)
        self.model = AlternatingLeastSquares(factors=factors, regularization=config["regularization"],
                alpha=config["library_alpha"], iterations=config["iterations"], random_state=config["seed"],
                dtype=np.float32, use_cg=config["use_cg"], num_threads=config["num_threads"], calculate_training_loss=True)
        self.model.cg_steps = config["cg_steps"]
        epochs = []
        self.model.fit(self.confidence, show_progress=False,
                       callback=lambda epoch, seconds, loss: epochs.append({"epoch": int(epoch), "seconds": float(seconds), "loss": float(loss)}))
        if self.model.user_factors.shape != (len(data.users), factors) or self.model.item_factors.shape != (len(data.items), factors):
            raise AssertionError("ALS matrix orientation drift")
        if not np.all(np.isfinite(self.model.user_factors)) or not np.all(np.isfinite(self.model.item_factors)):
            raise ValueError("Nonfinite ALS factors")
        self.diagnostics.update({"method": "implicit_als", "package_version": implicit.__version__, "factors": factors,
                "confidence": config["confidence"], "matrix_shape_users_items": list(self.confidence.shape),
                "confidence_min": float(self.confidence.data.min()) if self.confidence.nnz else None,
                "confidence_max": float(self.confidence.data.max()) if self.confidence.nnz else None,
                "epochs": epochs, "user_factor_variance": float(np.var(self.model.user_factors)),
                "item_factor_variance": float(np.var(self.model.item_factors)), "repeat_filter": False})

    def native(self, user, candidates):
        index = self.data.user_index.get(user)
        if index is None:
            return {}
        known = [(item, self.data.item_index[item]) for item in candidates if item in self.data.item_index]
        if not known:
            return {}
        scores = self.model.item_factors[[i for _, i in known]] @ self.model.user_factors[index]
        return {item: float(score) for (item, _), score in zip(known, scores)}


def required_components(specs):
    names = set()
    for spec in specs:
        name, kind = spec["name"], spec["kind"]
        if kind in {"hybrid", "hybrid_als", "history_fusion"}:
            names.update(["history_seconds", "history_watches", "history_recent"])
            if kind in {"hybrid", "hybrid_als"}:
                names.update(["ppr_v2", "channel_time"])
            if kind == "hybrid_als":
                names.add("als32")
        else:
            names.add(name)
    return names


class Family:
    """One cutoff's shared fitted components; nothing is shared across cutoffs."""
    def __init__(self, train, metadata, recency, state, exposure, items, config, specs):
        self.config, self.items, self.pool = config, list(items), set(items)
        self.models, self.fit_seconds, self.failures, self.native_diagnostics = {}, {}, {}, {}
        self.predict_seconds = {}
        started = time.perf_counter()
        self.data, seeds, restart, seed_diag = _prepare(train, "seconds", .5)
        self.shared_data_seconds = time.perf_counter() - started
        self.fallback_order = sorted(self.pool, key=lambda item: (-self.data.popularity.get(item, 0), item))
        required = required_components(specs)
        graph = None
        self.shared_graph_seconds = 0.
        if required & {"original_ppr_0.5", "ppr_v2"}:
            started = time.perf_counter()
            graph = _pruned_transition(self.data, 100)
            self.shared_graph_seconds = time.perf_counter() - started
        for name in sorted(required):
            start = time.perf_counter()
            print("Fitting v3 component: " + name, flush=True)
            try:
                if name == "popularity":
                    model = PopularityModel(self.data)
                elif name == "content":
                    model = ContentModel(self.data, metadata)
                elif name in {"original_ppr_0.5", "ppr_v2"}:
                    graph_state = graph if name == "original_ppr_0.5" else (graph[0], graph[1], restart, graph[3], graph[4])
                    model = PPRModel(self.data, alpha=.5 if name == "original_ppr_0.5" else .25, graph_state=graph_state)
                    if name == "ppr_v2":
                        model.diagnostics.update(seed_diag)
                        model.diagnostics["method"] = "v2_equivalent_ppr_sqrt_seconds"
                elif name.startswith("history_"):
                    field = {"history_seconds": "seconds", "history_watches": "watches", "history_recent": "recent_seconds"}[name]
                    model = SignalModel(self.data, recency if name == "history_recent" else train, field)
                elif name == "channel_time":
                    model = ChannelModel(self.data, state, exposure, items)
                elif name in {"ppr_cosine", "ppr_lift"}:
                    kind = name.split("_")[1]
                    graph_state = normalized_graph(self.data, restart, kind, config["graph"])
                    model = PPRModel(self.data, alpha=.25, graph_state=graph_state)
                    model.diagnostics.update(seed_diag)
                    model.diagnostics["method"] = name
                elif name in {"als32", "als64"}:
                    model = ALSModel(self.data, int(name[3:]), config["als"])
                else:
                    raise ValueError(name)
                if time.perf_counter() - start > config["resources"]["max_method_wall_seconds"]:
                    raise TimeoutError("Component exceeded frozen wall-time limit")
                self.models[name] = model
            except Exception as error:
                if name not in {"als32", "als64"}:
                    raise
                import traceback
                self.failures[name] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
            self.fit_seconds[name] = time.perf_counter() - start

    def lists(self, user):
        result = {}
        for name, model in self.models.items():
            started = time.perf_counter()
            scores = native_scores(model, user, self.pool)
            values = np.asarray(list(scores.values()), dtype=np.float64)
            self.native_diagnostics.setdefault(name, {})[user] = {
                "scored_candidates": len(values), "positive_candidates": int(np.count_nonzero(values > 0)),
                "minimum": float(values.min()) if len(values) else None,
                "maximum": float(values.max()) if len(values) else None,
                "variance": float(np.var(values)) if len(values) else None}
            result[name] = ordered_native(scores, self.config["fusion"]["native_top_n"])
            self.predict_seconds[name] = self.predict_seconds.get(name, 0.) + time.perf_counter() - started
        return result

    def compose(self, spec, lists):
        kind, name = spec["kind"], spec["name"]
        if kind not in {"history_fusion", "hybrid", "hybrid_als"}:
            if name not in lists:
                raise RuntimeError("Unavailable component: " + name)
            return dict(lists[name]), {}, set(item for item, _ in lists[name])
        history_lists = {key: lists[key] for key in ["history_seconds", "history_watches", "history_recent"]}
        history_scores, history_contributions = rrf(history_lists, dict(zip(history_lists, self.config["fusion"]["history_weights"])),
                                                      self.config["fusion"]["rrf_constant"], self.pool)
        history_rank = ordered_native(history_scores, 150)
        if kind == "history_fusion":
            return history_scores, history_contributions, set(history_scores)
        union = set(item for records in [lists["ppr_v2"], lists["channel_time"], *history_lists.values()]
                    for item, _ in records) & self.pool
        component_lists = {"ppr_v2": lists["ppr_v2"], "personal_history_fusion": history_rank, "channel_time": lists["channel_time"]}
        if kind == "hybrid_als":
            if "als32" not in lists:
                raise RuntimeError("Unavailable component: als32")
            union.update(item for item, _ in lists["als32"])
            component_lists["als32"] = lists["als32"]
            weights = {"ppr_v2": .5 * spec["weights"][0], "als32": .5 * spec["weights"][0],
                       "personal_history_fusion": spec["weights"][1], "channel_time": spec["weights"][2]}
        else:
            weights = dict(zip(component_lists, spec["weights"]))
        if len(union) > self.config["fusion"]["max_union_items"]:
            raise AssertionError("Retrieval union exceeds frozen capacity")
        scores, contributions = rrf(component_lists, weights, self.config["fusion"]["rrf_constant"], union)
        return scores, contributions, union

    def recommend_from(self, user, spec, lists):
        scores, contributions, union = self.compose(spec, lists)
        context_users = self.models["channel_time"].state.user_index if "channel_time" in self.models else {}
        cold = user not in self.data.user_index and user not in context_users
        # Cache the same common popularity ordering; do not sort the full pool
        # for every model/user after a full native list already exists.
        records = [{"item_id": item, "score": score, "reason": spec["name"]}
                   for item, score in ordered_native(scores, 5)]
        selected = {row["item_id"] for row in records}
        reason = "cold_user:training_popularity" if cold else (
            "no_positive_model_score:training_popularity" if not records else "short_model_list:training_popularity")
        if len(records) < 5:
            for item in self.fallback_order:
                if item not in selected:
                    records.append({"item_id": item, "score": float(self.data.popularity.get(item, 0)), "reason": reason})
                    selected.add(item)
                    if len(records) == 5:
                        break
        if len(records) != 5:
            raise ValueError("Fewer than five eligible candidates")
        return records, contributions, union


class SelectedModel:
    def __init__(self, family, specification):
        self.family, self.specification = family, specification

    def recommend(self, user):
        return self.family.recommend_from(user, self.specification, self.family.lists(user))[0]

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            pickle.dump(self, stream, protocol=pickle.HIGHEST_PROTOCOL)
