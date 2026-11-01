"""Independent dense normal-equation and cold-factor checks."""
from types import SimpleNamespace
import pickle
import unittest

import numpy as np
from scipy import sparse
from sweettv.models import _TrainingData
from sweettv.target040_inductive import InductiveMF


class InductiveTests(unittest.TestCase):
    def test_dense_ridge_orientation_known_factor_identity_and_cold_prediction(self):
        data = _TrainingData([{"user_id":"u","item_id":i,"seconds":120.,"watches":1} for i in [1,2,3]])
        base = SimpleNamespace(data=data,item_factors=np.array([[2.,0.],[0.,2.],[1.,1.]]),user_factors=np.array([[.7,.3]]))
        history = np.array([[1.,0.],[0.,1.],[1.,1.]])
        future = np.array([[99.,99.],[2.,0.],[0.,0.]])
        spec = {"name":"inductive","ridge_alpha":1.,"cold_scale":.25}
        config = {"tol":1e-12,"max_iter":1000,"features":"Synthetic two-slot exposure"}
        model = InductiveMF(base,sparse.csr_matrix(history),[1,2,3],sparse.csr_matrix(future),[1,9,10],spec,config)
        coefficient = np.linalg.solve(history.T@history+np.eye(2),history.T@base.item_factors)
        np.testing.assert_allclose(model.ridge.coef_.T,coefficient,atol=1e-10)
        np.testing.assert_array_equal(model.candidate_factors[0],base.item_factors[0])
        np.testing.assert_allclose(model.candidate_factors[1],np.maximum(future[1]@coefficient,0)*.25,atol=1e-10)
        self.assertEqual(model.native("cold",{1,9}),{})
        self.assertEqual(model.native("u",{10}),{10:0.})
        scores = model.native("u",{1,9,10,999});self.assertNotIn(999,scores)
        self.assertEqual(scores[1],1.4)
        self.assertAlmostEqual(scores[9],float(model.candidate_factors[1]@base.user_factors[0]))
        self.assertEqual(pickle.loads(pickle.dumps(model)).native("u",{1,9,10,999}),scores)
        reused = InductiveMF(base,history,[1,2,3],future,[1,9,10],spec,config,fitted_ridge=model.ridge)
        self.assertEqual(reused.native("u",{1,9,10}),model.native("u",{1,9,10}))
        with self.assertRaises(ValueError):InductiveMF(base,history,[3,2,1],future,[1,9,10],spec,config)
        with self.assertRaises(ValueError):InductiveMF(base,history,[1,2,3],future[:1],[1,9,10],spec,config)


if __name__=="__main__":unittest.main()
