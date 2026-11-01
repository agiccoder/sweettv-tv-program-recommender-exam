"""Cold completion keeps known intensities and aligns against known MF mass."""
import unittest
from types import SimpleNamespace
from sweettv.models import _TrainingData
from sweettv.target040_cold import ColdCompletion

class ColdCompletionTests(unittest.TestCase):
    def test_known_identity_cold_mass_alignment_subset_and_no_padding(self):
        data=_TrainingData([{'user_id':'u','item_id':i,'seconds':120.,'watches':1} for i in [1,2]])
        def fixed(values):return SimpleNamespace(data=data,native=lambda user,pool:{i:v for i,v in values.items() if i in pool} if user=='u' else {})
        model=ColdCompletion(fixed({1:.7,2:.3}),fixed({1:.2,2:.3,9:.5}),[1,2,9],{'name':'cold'})
        self.assertEqual(model.native('u',{1,2,9}),{1:.7,2:.3,9:1.})
        self.assertEqual(model.native('u',{9}),{9:1.})
        self.assertEqual(model.native('cold',{1,9}),{})
        empty=ColdCompletion(fixed({1:.7}),fixed({9:.5}),[1,2,9],{'name':'cold'})
        self.assertEqual(empty.native('u',{1,2,9}),{1:.7})
        with self.assertRaises(ValueError):model.native('u',{99})

if __name__=='__main__':unittest.main()
