"""Training-only recommendation models for the SWEET.TV exam.

The caller must pass only interactions and schedule metadata available before its
training cutoff. No recommendation call fits a transform or consumes metadata.
Repeat recommendations are allowed consistently for every model.

Popularity is the number of distinct training viewers of a programme. Content
uses field-prefixed category, genre, production-year, director and actor tokens
(never titles), binary term frequencies, smoothed IDF, and L2 item vectors. A
user profile is the L2-normalized sum of item vectors weighted by log1p of the
user's aggregated overlap seconds. Its score is cosine similarity.

For personalized PageRank (PPR), X[u,i] = log1p(seconds[u,i]) divided by its
user-row sum. A = X.T @ X, with diagonal removed, gives every user's outer
product total mass at most one, limiting domination by heavy histories. Each
row retains at most 100 strongest neighbours (weight descending, item ID
ascending), then is normalized into a transition matrix P. For normalized user
history v, iterate p' = alpha * (P.T @ p + dangling_mass * v) + (1-alpha)*v.
Stop when the L1 iterate difference is <= 1e-8, with at most 300 iterations.
Only historical interaction items are graph nodes. Unseen target items have no
learned representation. Nonpositive/missing scores and cold users use the same
training-popularity fallback, restricted to the requested candidate pool.

The module contains no data-loading shortcuts or external service calls. Tests
use explicitly synthetic fixtures; diagnostics are measurements of fitted
state and actual recommendation calls, not validation-quality claims.
"""

from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional
import math
import numbers
import pickle
import re
import time
import unicodedata

import numpy as np
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize


CONTENT_FIELDS = (
    "tv_show_category", "tv_show_genre_1", "tv_show_genre_2",
    "tv_show_genre_3", "year_of_production", "director", "actors",
)
_ALIASES = {
    "tv_show_category": ("category",),
    "tv_show_genre_1": ("genre1", "genre_1"),
    "tv_show_genre_2": ("genre2", "genre_2"),
    "tv_show_genre_3": ("genre3", "genre_3"),
    "year_of_production": ("year", "production_year"),
    "director": (), "actors": (),
}


def _records(interactions: Any) -> Iterable[Mapping[str, Any]]:
    return interactions.to_dict("records") if hasattr(interactions, "to_dict") else interactions


class _TrainingData:
    def __init__(self, interactions: Any):
        aggregated = {}
        input_rows = 0
        for row in _records(interactions):
            input_rows += 1
            user, item = row["user_id"], row["item_id"]
            if not isinstance(user, str):
                raise ValueError("user_id must be a string; IDs must never pass through float")
            if not isinstance(item, numbers.Integral) or isinstance(item, bool) or item <= 0:
                raise ValueError("item_id must be a positive integer; 0 is unlabelled airtime")
            seconds, watches = float(row["seconds"]), float(row["watches"])
            if not math.isfinite(seconds) or seconds < 0 or not math.isfinite(watches) or watches < 0 or not watches.is_integer():
                raise ValueError("seconds must be finite nonnegative; watches a nonnegative integer")
            if seconds <= 0:
                continue
            key = (user, int(item))
            previous = aggregated.get(key, (0.0, 0))
            aggregated[key] = (previous[0] + seconds, previous[1] + int(watches))
        self.users = sorted({user for user, _ in aggregated})
        self.items = np.asarray(sorted({item for _, item in aggregated}), dtype=np.int64)
        self.user_index = {user: i for i, user in enumerate(self.users)}
        self.item_index = {int(item): i for i, item in enumerate(self.items)}
        rows, cols, values = [], [], []
        for (user, item), (seconds, _) in sorted(aggregated.items()):
            rows.append(self.user_index[user])
            cols.append(self.item_index[item])
            values.append(math.log1p(seconds))
        self.weights = sparse.csr_matrix((values, (rows, cols)), shape=(len(self.users), len(self.items)), dtype=np.float64)
        self.popularity = {int(item): int(count) for item, count in zip(self.items, self.weights.getnnz(axis=0))}
        self.diagnostics = {
            "input_interaction_rows": input_rows,
            "training_users": len(self.users), "training_items": len(self.items),
            "unique_user_item_interactions": self.weights.nnz,
            "history_weight": "log1p(sum overlap seconds per user and item)",
            "fit_boundary": "caller-supplied training interactions only; no candidate-time fitting",
        }


class _BaseModel:
    def __init__(self, data: _TrainingData):
        self.data = data
        self.diagnostics = dict(data.diagnostics)
        self.diagnostics.update({
            "repeat_policy": "allow previously watched programmes",
            "tie_break": "score descending, programme ID ascending",
            "fallback": "distinct training-viewer popularity restricted to target candidates; ties by ID",
            "recommendation_calls": 0,
            "fallback_items_returned": 0,
        })

    def save(self, path: Any) -> None:
        """Persist actual fitted state with pickle; load only trusted local files."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as stream:
            pickle.dump(self, stream, protocol=pickle.HIGHEST_PROTOCOL)

    def _candidates(self, candidates: Iterable[int], k: int) -> List[int]:
        if not isinstance(k, numbers.Integral) or k <= 0:
            raise ValueError("k must be a positive integer")
        result = set()
        for item in candidates:
            if not isinstance(item, numbers.Integral) or isinstance(item, bool) or item <= 0:
                raise ValueError("all candidates must be positive integer programme IDs")
            result.add(int(item))
        if len(result) < max(5, k):
            raise ValueError("at least five distinct candidates, and at least k, are required")
        return sorted(result)

    def _rank(self, candidates: List[int], scores: Mapping[int, float], k: int, reason: str, cold: bool = False) -> List[dict]:
        ranked = [(item, float(scores.get(item, 0.0))) for item in candidates]
        ranked = [(item, score) for item, score in ranked if math.isfinite(score) and score > 0]
        ranked.sort(key=lambda x: (-x[1], x[0]))
        result = [{"item_id": item, "score": score, "reason": reason} for item, score in ranked[:k]]
        selected = {row["item_id"] for row in result}
        fallback_reason = "cold_user:training_popularity" if cold else (
            "no_positive_model_score:training_popularity" if not result else "short_model_list:training_popularity")
        fallback = sorted((item for item in candidates if item not in selected), key=lambda item: (-self.data.popularity.get(item, 0), item))
        for item in fallback[:k-len(result)]:
            result.append({"item_id": item, "score": float(self.data.popularity.get(item, 0)), "reason": fallback_reason})
            self.diagnostics["fallback_items_returned"] += 1
        self.diagnostics["recommendation_calls"] += 1
        return result


class PopularityModel(_BaseModel):
    def __init__(self, data: _TrainingData):
        super().__init__(data)
        self.diagnostics.update({
            "method": "popularity",
            "definition": "number of distinct users with positive training overlap seconds per item",
            "training_item_score_variance": float(np.var(list(data.popularity.values()))) if data.popularity else 0.0,
        })

    def recommend(self, user_id: str, candidates: List[int], k: int = 5) -> List[dict]:
        pool = self._candidates(candidates, k)
        # Zero-count programmes remain legitimate candidates, deterministically
        # appended by the same fallback used for personalized models.
        return self._rank(pool, self.data.popularity, k, "training_popularity", user_id not in self.data.user_index)


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, numbers.Real) and not math.isfinite(float(value)):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    return "" if text in {"", "nan", "none", "null", "<na>"} else re.sub(r"\s+", " ", text)


def _feature_dict(metadata: Mapping[str, Any], fields: Iterable[str]) -> Dict[str, float]:
    features = {}
    for field in fields:
        value = metadata.get(field)
        if value is None:
            for alias in _ALIASES.get(field, ()):
                if alias in metadata:
                    value = metadata[alias]
                    break
        values = value if isinstance(value, (list, tuple, set)) else [value]
        for value in values:
            token = _text(value)
            if field == "year_of_production" and token:
                try:
                    number = float(token)
                    if not math.isfinite(number) or number <= 0:
                        continue
                    token = str(int(number))
                except ValueError:
                    pass
            # Split before whitespace normalization so a newline remains a
            # genuine person separator rather than becoming an ordinary space.
            parts = re.split(r"[,;|\n]+", str(value)) if field in {"actors", "director"} else [token]
            for part in parts:
                part = _text(part)
                if part:
                    features[field + "=" + part] = 1.0
    return features


class ContentModel(_BaseModel):
    def __init__(self, data: _TrainingData, metadata: Mapping[int, Mapping[str, Any]], content_fields: Optional[Iterable[str]] = None):
        started = time.perf_counter()
        super().__init__(data)
        self.fields = tuple(CONTENT_FIELDS if content_fields is None else content_fields)
        if any(field not in CONTENT_FIELDS for field in self.fields):
            raise ValueError("content_fields must use permitted canonical non-title fields")
        self.metadata_items = np.asarray(sorted(int(item) for item in metadata if int(item) > 0), dtype=np.int64)
        self.metadata_index = {int(item): i for i, item in enumerate(self.metadata_items)}
        feature_rows = [_feature_dict(metadata[int(item)], self.fields) for item in self.metadata_items]
        self.vectorizer = None
        self.tfidf = None
        if any(feature_rows):
            self.vectorizer = DictVectorizer(sparse=True, dtype=np.float64)
            counts = self.vectorizer.fit_transform(feature_rows)
            self.tfidf = TfidfTransformer(norm="l2", use_idf=True, smooth_idf=True, sublinear_tf=False)
            self.item_vectors = self.tfidf.fit_transform(counts).tocsr()
        else:
            self.item_vectors = sparse.csr_matrix((len(self.metadata_items), 0), dtype=np.float64)
        rr, cc = [], []
        for i, item in enumerate(data.items):
            j = self.metadata_index.get(int(item))
            if j is not None:
                rr.append(i)
                cc.append(j)
        mapping = sparse.csr_matrix((np.ones(len(rr)), (rr, cc)), shape=(len(data.items), len(self.metadata_items)))
        self.profiles = normalize(data.weights @ mapping @ self.item_vectors, norm="l2", axis=1).tocsr() if self.item_vectors.shape[1] and len(data.users) else sparse.csr_matrix((len(data.users), self.item_vectors.shape[1]))
        sample_variances = []
        if self.item_vectors.shape[0] and self.profiles.shape[1]:
            for i in range(min(32, self.profiles.shape[0])):
                scores = (self.item_vectors @ self.profiles.getrow(i).T).toarray().ravel()
                sample_variances.append(float(np.var(scores)))
        self.diagnostics.update({
            "method": "content_cosine", "content_fields": list(self.fields),
            "training_metadata_items": len(self.metadata_items),
            "vocabulary_size": self.item_vectors.shape[1],
            "item_vector_nonzeros": self.item_vectors.nnz,
            "nonempty_profiles": int(np.count_nonzero(self.profiles.getnnz(axis=1))),
            "idf": "log((1+n_training_metadata_items)/(1+document_frequency))+1",
            "term_frequency": "binary field-prefixed tokens; actors/directors split on comma, semicolon, pipe or newline",
            "score_variance_training_sample": sample_variances,
            "score_variance_sample_definition": "item cosine scores for first <=32 lexicographically sorted training users",
            "metadata_boundary": "all metadata and vocabulary supplied by caller from training schedule only",
            "fit_seconds": time.perf_counter() - started,
        })

    def recommend(self, user_id: str, candidates: List[int], k: int = 5) -> List[dict]:
        pool = self._candidates(candidates, k)
        user = self.data.user_index.get(user_id)
        cold = user is None or self.profiles.getrow(user).nnz == 0
        scores = {}
        if not cold:
            known = [(item, self.metadata_index[item]) for item in pool if item in self.metadata_index]
            if known:
                values = (self.item_vectors[[index for _, index in known]] @ self.profiles.getrow(user).T).toarray().ravel()
                scores = {item: float(score) for (item, _), score in zip(known, values)}
        return self._rank(pool, scores, k, "content_cosine", cold)


def _pruned_transition(data: _TrainingData, max_neighbors: int):
    if max_neighbors < 1:
        raise ValueError("max_neighbors must be positive")
    # Explicit capacity bound plus row-block sparse multiplication prevents a
    # prolific user from materializing an uncontrolled full item-pair matrix.
    # The estimate covers sparse product payload, not total process RSS or the
    # multiplication implementation's workspace, which the caller measures.
    number_items = len(data.items)
    maximum_items, block_budget_bytes = 100000, 64 * 1024 * 1024
    if number_items > maximum_items:
        raise ValueError("PPR graph exceeds the explicit 100000 training-item capacity bound")
    block_size = min(128, max(1, block_budget_bytes // max(16 * number_items, 1)))
    row_sums = np.asarray(data.weights.sum(axis=1)).ravel()
    weights = sparse.diags(np.divide(1.0, row_sums, out=np.zeros_like(row_sums), where=row_sums > 0)) @ data.weights
    weights = weights.tocsr()
    transposed = weights.T.tocsr()
    before, columns, values, indptr = 0, [], [], [0]
    # Each product contains <=block_size rows; prune it before constructing the
    # next one. The only item-row loop selects existing sparse neighbours; no
    # Python user-pair enumeration or full item-item matrix is constructed.
    for block_start in range(0, number_items, block_size):
        block = (transposed[block_start:block_start+block_size] @ weights).tocsr()
        for offset in range(block.shape[0]):
            begin, end = block.indptr[offset:offset+2]
            indices, scores = block.indices[begin:end], block.data[begin:end]
            keep = (indices != block_start+offset) & (scores > 0)
            indices, scores = indices[keep], scores[keep]
            before += len(indices)
            order = np.lexsort((data.items[indices], -scores))[:max_neighbors]
            columns.append(indices[order].astype(np.int32, copy=False))
            values.append(scores[order])
            indptr.append(indptr[-1] + len(order))
    adjacency = sparse.csr_matrix((np.concatenate(values) if values else np.empty(0),
                                   np.concatenate(columns) if columns else np.empty(0, dtype=np.int32),
                                   np.asarray(indptr, dtype=np.int64)),
                                  shape=(number_items, number_items), dtype=np.float64)
    sums = np.asarray(adjacency.sum(axis=1)).ravel()
    transition = (sparse.diags(np.divide(1.0, sums, out=np.zeros_like(sums), where=sums > 0)) @ adjacency).tocsr()
    transition.sort_indices()
    graph_diagnostics = {
        "graph_capacity_maximum_items": maximum_items,
        "graph_multiplication": "sparse item-row blocks; immediate deterministic top-neighbour pruning",
        "graph_product_block_rows": block_size,
        "graph_product_payload_budget_bytes": block_budget_bytes,
        "graph_product_payload_upper_estimate_bytes": min(block_size, number_items) * number_items * 16,
        "graph_retained_payload_upper_estimate_bytes": number_items * min(max_neighbors, max(0, number_items-1)) * 16 + (number_items+1) * 8,
        "graph_memory_estimate_scope": "matrix payload only, excludes inputs and sparse multiplication workspace; caller measures RSS",
    }
    return transition, sums == 0, weights, before, graph_diagnostics


class PPRModel(_BaseModel):
    def __init__(self, data: _TrainingData, alpha: float = .85, max_neighbors: int = 100, tolerance: float = 1e-8, max_iterations: int = 300, graph_state: Any = None):
        started = time.perf_counter()
        super().__init__(data)
        if not 0 < alpha < 1 or tolerance <= 0 or max_iterations < 1:
            raise ValueError("require 0 < alpha < 1, positive tolerance and iteration cap")
        self.alpha, self.tolerance, self.max_iterations = alpha, tolerance, max_iterations
        self.transition, self.dangling, self.restart_rows, before, graph_diagnostics = _pruned_transition(data, max_neighbors) if graph_state is None else graph_state
        self.transition_t = self.transition.T.tocsr()
        self._cache = OrderedDict()
        sums = np.asarray(self.transition.sum(axis=1)).ravel()
        self.diagnostics.update({
            "method": "personalized_pagerank", "alpha_link_following": alpha,
            "restart_probability": 1-alpha, "tolerance_l1": tolerance,
            "max_iterations": max_iterations, "max_neighbors": max_neighbors,
            "graph_nodes": len(data.items), "graph_edges_before_pruning": before,
            "graph_edges": self.transition.nnz, "dangling_nodes": int(self.dangling.sum()),
            "maximum_nondangling_row_sum_error": float(np.max(np.abs(sums[~self.dangling]-1))) if np.any(~self.dangling) else 0.0,
            "graph_weight": "X.T @ X; X=log1p(seconds), independently L1-normalized per user; diagonal removed",
            "dangling_policy": "redistribute all dangling mass to the same user restart distribution",
            "walk_calls": 0, "converged_walks": 0, "nonconverged_walks": 0,
            "max_observed_iterations": 0, "max_observed_l1_residual": 0.0,
            "max_observed_mass_error": 0.0, "score_variance_min": None,
            "score_variance_max": None, "per_user_walk": {},
            "fit_seconds": time.perf_counter() - started,
        })
        self.diagnostics.update(graph_diagnostics)

    def _walk(self, user_id: str) -> np.ndarray:
        if user_id in self._cache:
            self._cache.move_to_end(user_id)
            return self._cache[user_id]
        user = self.data.user_index[user_id]
        restart = self.restart_rows.getrow(user).toarray().ravel()
        p = restart.copy()
        residual = math.inf
        for iteration in range(1, self.max_iterations+1):
            next_p = self.alpha * (self.transition_t @ p + p[self.dangling].sum() * restart) + (1-self.alpha) * restart
            residual = float(np.abs(next_p-p).sum())
            p = next_p
            if residual <= self.tolerance:
                break
        converged = residual <= self.tolerance
        variance = float(np.var(p)) if len(p) else 0.0
        mass_error = abs(float(p.sum())-1.0)
        diagnostics = self.diagnostics
        diagnostics["walk_calls"] += 1
        diagnostics["converged_walks" if converged else "nonconverged_walks"] += 1
        diagnostics["max_observed_iterations"] = max(diagnostics["max_observed_iterations"], iteration)
        diagnostics["max_observed_l1_residual"] = max(diagnostics["max_observed_l1_residual"], residual)
        diagnostics["max_observed_mass_error"] = max(diagnostics["max_observed_mass_error"], mass_error)
        diagnostics["score_variance_min"] = variance if diagnostics["score_variance_min"] is None else min(variance, diagnostics["score_variance_min"])
        diagnostics["score_variance_max"] = variance if diagnostics["score_variance_max"] is None else max(variance, diagnostics["score_variance_max"])
        diagnostics["per_user_walk"][user_id] = {"iterations": iteration, "l1_residual": residual, "mass": float(p.sum()), "converged": converged, "score_variance": variance}
        self._cache[user_id] = p
        if len(self._cache) > 16:
            self._cache.popitem(last=False)
        return p

    def recommend(self, user_id: str, candidates: List[int], k: int = 5) -> List[dict]:
        pool = self._candidates(candidates, k)
        cold = user_id not in self.data.user_index
        scores = {}
        if not cold:
            p = self._walk(user_id)
            scores = {item: float(p[self.data.item_index[item]]) for item in pool if item in self.data.item_index}
        return self._rank(pool, scores, k, "personalized_pagerank", cold)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_cache"] = OrderedDict()
        return state


def fit_popularity(interactions: Any, metadata: Optional[Mapping] = None) -> PopularityModel:
    return PopularityModel(_TrainingData(interactions))


def fit_content(interactions: Any, metadata: Mapping[int, Mapping], *, content_fields: Optional[Iterable[str]] = None) -> ContentModel:
    return ContentModel(_TrainingData(interactions), metadata, content_fields)


def fit_ppr(interactions: Any, metadata: Optional[Mapping] = None, *, alpha: float = .85, max_neighbors: int = 100, tolerance: float = 1e-8, max_iterations: int = 300) -> PPRModel:
    return PPRModel(_TrainingData(interactions), alpha, max_neighbors, tolerance, max_iterations)


def fit_models(interactions: Any, metadata: Mapping[int, Mapping], *, alpha: float = .85, content_fields: Optional[Iterable[str]] = None) -> Dict[str, _BaseModel]:
    """Fit the common baseline and two qualifying families on one training set.

    Metadata is explicitly the caller's training-period schedule dictionary.
    Candidates passed at prediction time supply identity/availability only.
    Use fit_ppr(..., alpha=.5) for the single additional documented alpha trial.
    """
    data = _TrainingData(interactions)
    return {"popularity": PopularityModel(data), "content": ContentModel(data, metadata, content_fields), "ppr": PPRModel(data, alpha=alpha)}
