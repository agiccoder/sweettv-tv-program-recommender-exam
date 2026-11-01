"""Schedule-conditioned item factors for previously unseen programme IDs.

This extends an implicit MF model; it does not fit future user preferences.
Known item factors are retained exactly. Ridge maps past broadcast descriptors
to past fitted item factors, then evaluates permitted target availability.
"""
import time

import numpy as np
from scipy import sparse
from sklearn.linear_model import Ridge

from .models import _BaseModel


class InductiveMF(_BaseModel):
    def __init__(self, base, history, history_items, future, items, specification, config, fitted_ridge=None):
        super().__init__(base.data)
        if list(history_items) != list(self.data.items):
            raise ValueError("Historical descriptor IDs must equal fitted MF item order")
        history, future = sparse.csr_matrix(history), sparse.csr_matrix(future)
        if history.shape[0] != len(history_items) or future.shape != (len(items),history.shape[1]):
            raise ValueError("Inductive descriptor orientation mismatch")
        if any(not np.isfinite(x.data).all() or np.any(x.data<0) for x in [history,future]):
            raise ValueError("Schedule descriptors must be finite nonnegative")
        self.base, self.history, self.future = base,history,future
        self.items = list(items);self.item_index = {item:i for i,item in enumerate(items)}
        self.alpha = float(specification["ridge_alpha"])
        self.cold_scale = float(specification["cold_scale"])
        if self.alpha<=0 or not 0<self.cold_scale<=1:
            raise ValueError("Invalid frozen ridge/shrinkage setting")
        started = time.perf_counter()
        self.ridge = fitted_ridge
        if self.ridge is None:
            self.ridge = Ridge(alpha=self.alpha,fit_intercept=False,copy_X=True,
                solver="lsqr",tol=config["tol"],max_iter=config["max_iter"])
            self.ridge.fit(history,base.item_factors)
        if self.ridge.coef_.shape != (base.item_factors.shape[1],history.shape[1]):
            raise ValueError("Ridge features/latent-factor orientation mismatch")
        self.raw_projection = np.asarray(future@self.ridge.coef_.T)
        self.candidate_factors = np.maximum(self.raw_projection,0)*self.cold_scale
        self.known_mask = np.array([item in self.data.item_index for item in items])
        for index,item in enumerate(items):
            if item in self.data.item_index:
                self.candidate_factors[index] = base.item_factors[self.data.item_index[item]]
        if not np.isfinite(self.candidate_factors).all() or np.any(self.candidate_factors<0):
            raise ValueError("Inductive item factors must be finite nonnegative")
        residual = history@self.ridge.coef_.T-base.item_factors
        gradient = history.T@residual+self.alpha*self.ridge.coef_.T
        denominator = max(float(np.linalg.norm(history.T@base.item_factors)),1.)
        self.diagnostics.update({"method":specification["name"],"qualifying_family":"implicit MF with schedule-conditioned cold-item factors",
            "objective":"min_B ||X_history B - V_MF||_F^2 + ridge_alpha ||B||_F^2; no intercept",
            "ridge_alpha":self.alpha,"cold_scale":self.cold_scale,"ridge_n_iter":self.ridge.n_iter_.tolist(),
            "ridge_fit_seconds":time.perf_counter()-started,"reused_same_cutoff_ridge":fitted_ridge is not None,
            "relative_normal_equation_residual":float(np.linalg.norm(gradient))/denominator,
            "training_factor_rmse":float(np.sqrt(np.mean(residual**2))),
            "known_candidate_count":int(self.known_mask.sum()),"cold_candidate_count":int((~self.known_mask).sum()),
            "negative_projected_factor_fraction":float(np.mean(self.raw_projection<0)),
            "known_factors_unchanged":True,"feature_definition":config["features"],
            "future_viewing_used":False,"future_content_attributes_used":False,"previously_viewed_items_filtered":False})

    def native(self,user,candidates):
        index = self.data.user_index.get(user)
        if index is None:return {}
        known = [(item,self.item_index[item]) for item in candidates if item in self.item_index]
        if not known:return {}
        values = np.einsum("ij,j->i",self.candidate_factors[[i for _,i in known]],self.base.user_factors[index],optimize=False)
        return {item:float(value) for (item,_),value in zip(known,values)}
