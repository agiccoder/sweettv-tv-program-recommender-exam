"""Shared-user implicit MF for paired count and duration intensities."""
import warnings

import numpy as np
from scipy import sparse
from sklearn.decomposition import NMF
from sweettv.models import _BaseModel


class JointIntensityFit:
    def __init__(self,data,frame,factors,config):
        self.data = data
        rows = [data.user_index[u] for u in frame.user_id]
        columns = [data.item_index[int(i)] for i in frame.item_id]
        blocks = []
        for signal in ["watches","seconds"]:
            matrix = sparse.csr_matrix((frame[signal].to_numpy(dtype=float),(rows,columns)),shape=data.weights.shape)
            totals = matrix.sum(axis=1).A.ravel()
            blocks.append(sparse.diags(np.divide(.5,totals,out=np.zeros_like(totals),where=totals>0))@matrix)
        self.training_intensity = sparse.hstack(blocks,format="csr")
        self.model = NMF(n_components=factors,init=config["init"],solver=config["solver"],beta_loss=config["beta_loss"],
            max_iter=config["max_iter"],tol=config["tol"],random_state=config["seed"],
            alpha_W=config["alpha_W"],alpha_H=config["alpha_H"],l1_ratio=config["l1_ratio"])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.user_factors = self.model.fit_transform(self.training_intensity)
        self.item_factors = self.model.components_.T.copy()
        if self.user_factors.shape != (len(data.users),factors) or self.item_factors.shape != (2*len(data.items),factors):
            raise ValueError("Joint-MF paired block orientation mismatch")
        if any(not np.isfinite(x).all() or np.any(x<0) for x in [self.user_factors,self.item_factors]):
            raise ValueError("Invalid joint nonnegative factors")
        self.diagnostics = {"factors":factors,"objective":"KL(.5 L1(watches)||U Vw.T) + KL(.5 L1(seconds)||U Vs.T) + sklearn dimension-scaled L2 regularization",
            "matrix_shape":list(self.training_intensity.shape),"block_order":["watches","seconds"],
            "actual_iterations":int(self.model.n_iter_),"sklearn_reconstruction_err":float(self.model.reconstruction_err_),
            "warnings":[str(w.message) for w in caught],"future_viewing_used":False}


class JointMFView(_BaseModel):
    def __init__(self,joint,view,items):
        super().__init__(joint.data)
        if view not in {"watches","seconds","mean"}:raise ValueError("Unknown joint-MF view")
        self.joint,self.view,self.pool = joint,view,set(items)
        self.diagnostics.update({"method":"joint_intensity_mf_"+view,"qualifying_family":"implicit matrix factorization",
            **joint.diagnostics,"score_view":view,"mean_formula":".5 L1_pool(U Vw.T) + .5 L1_pool(U Vs.T); zero view contributes0; before top50",
            "unknown_ID_policy":"No invented factor; common fallback","previously_viewed_items_filtered":False})

    def native(self,user,candidates):
        index = self.data.user_index.get(user)
        if index is None:return {}
        if not set(candidates)<=self.pool:raise ValueError("Joint MF cannot expand its frozen pool")
        full = self.pool if self.view=="mean" else candidates
        known = [(item,self.data.item_index[item]) for item in full if item in self.data.item_index]
        if not known:return {}
        indices = [j for _,j in known];nitems = len(self.data.items)
        first = np.einsum("ij,j->i",self.joint.item_factors[indices],self.joint.user_factors[index],optimize=False)
        second = np.einsum("ij,j->i",self.joint.item_factors[[j+nitems for j in indices]],self.joint.user_factors[index],optimize=False)
        if self.view=="watches":values = first
        elif self.view=="seconds":values = second
        else:
            values = np.zeros(len(known))
            for vector in [first,second]:
                total = float(vector.sum())
                if total>0:values += .5*vector/total
        return {item:float(value) for (item,_),value in zip(known,values) if item in candidates}
