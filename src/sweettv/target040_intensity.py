"""Normalized implicit-MF/PPR ranking mixtures with complete-pool scaling."""
import math

from .models import _BaseModel
from .v3_models import native_scores


class IntensityMixture(_BaseModel):
    """Preserve native intensity ratios; never normalize a truncated top list."""
    def __init__(self, models, items, weights, specification):
        if len(models) != len(weights) or not models:
            raise ValueError("Mixture requires one weight per component")
        if any(not math.isfinite(w) or w < 0 for w in weights) or not math.isclose(sum(weights), 1.):
            raise ValueError("Mixture weights must be finite, nonnegative and sum to one")
        super().__init__(models[0].data)
        self.components = models
        self.weights = list(weights)
        self.pool = set(items)
        self.diagnostics.update({"method": specification["name"], "qualifying_family": specification["qualifying_family"],
            "formula": "sum_k w_k*s_k(u,i)/sum_j_in_full_pool(s_k(u,j)); absent or zero component contributes0",
            "normalization_scope": "Complete common candidate pool, before native top50 or fallback",
            "component_weights_renormalized": False, "calibrated_future_probability_claimed": False,
            "future_viewing_used": False, "repeats_allowed": True})

    def native(self, user, candidates):
        if not set(candidates) <= self.pool:
            raise ValueError("Mixture cannot expand its frozen common pool")
        result = {}
        for model, weight in zip(self.components, self.weights):
            scores = {item:float(score) for item,score in native_scores(model, user, self.pool).items()
                      if math.isfinite(score) and score > 0}
            total = math.fsum(scores.values())
            if not total or not weight:
                continue
            for item in candidates:
                score = scores.get(item, 0.)
                if score > 0:
                    result[item] = result.get(item, 0.) + weight*score/total
        return result
