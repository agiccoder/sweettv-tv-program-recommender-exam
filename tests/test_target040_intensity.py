"""Independent hand checks for full-pool intensity mixtures."""
from types import SimpleNamespace
import pickle
import unittest

from sweettv.models import _TrainingData
from sweettv.target040_intensity import IntensityMixture


class FixedNative:
    def __init__(self, data, scores):
        self.data, self.scores = data, scores

    def native(self, user, candidates):
        return {i:s for i,s in self.scores.items() if i in candidates} if user == "u" else {}


class IntensityTests(unittest.TestCase):
    def test_full_pool_normalization_missing_component_repeats_and_roundtrip(self):
        data = _TrainingData([{"user_id":"u","item_id":1,"seconds":120.,"watches":1}])
        first, second = FixedNative(data,{1:6.,2:2.,3:2.}), FixedNative(data,{2:1.,3:3.})
        spec = {"name":"mixture","qualifying_family":"implicit MF/PPR hybrid"}
        model = IntensityMixture([first,second],[1,2,3],[.75,.25],spec)
        scores = model.native("u",{1,2,3})
        for i,expected in {1:.45,2:.2125,3:.3375}.items():
            self.assertAlmostEqual(scores[i],expected)
        self.assertEqual(model.native("u",{1}),{1:scores[1]})
        self.assertEqual(model.native("cold",{1,2,3}),{})
        self.assertEqual(pickle.loads(pickle.dumps(model)).native("u",{1,2,3}),scores)
        missing = IntensityMixture([first,FixedNative(data,{})],[1,2,3],[.75,.25],spec)
        self.assertAlmostEqual(sum(missing.native("u",{1,2,3}).values()),.75)
        with self.assertRaises(ValueError):model.native("u",{4})
        for weights in [[1.],[.8,.3],[-.1,1.1],[float("nan"),1.]]:
            with self.assertRaises(ValueError):IntensityMixture([first,second],[1,2,3],weights,spec)
        invalid = IntensityMixture([FixedNative(data,{1:2.,2:-2.,3:float("nan")})],[1,2,3],[1.],spec)
        self.assertEqual(invalid.native("u",{1,2,3}),{1:1.})


if __name__ == "__main__":
    unittest.main()
