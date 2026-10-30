"""Synthetic fixtures only; no competition measurements."""
import unittest
from sweettv.data import channel_segments, overlap_segments, union_intervals

class IntervalTests(unittest.TestCase):
    def test_half_open_and_multiple_airings(self):
        seg = channel_segments([(0,100,1),(100,200,2),(200,300,3)])
        got = list(overlap_segments([(50,250)],seg))
        self.assertEqual([(x[2][2],x[4]-x[3]) for x in got],[(1,50),(2,100),(3,50)])
        self.assertEqual(list(overlap_segments([(300,400)],seg)),[])
        self.assertEqual(list(overlap_segments([(-50,0)],seg)),[])

    def test_overlapping_views_and_duplicates(self):
        seg = channel_segments([(0,100,1),(0,100,1),(20,80,1)])
        got = list(overlap_segments([(10,50),(40,90),(10,50)],seg))
        self.assertEqual(sum(x[4]-x[3] for x in got),80)
        self.assertEqual(len(seg),1)

    def test_fragment_aggregation(self):
        got = list(overlap_segments([(0,70),(100,170)],channel_segments([(0,200,1)])))
        self.assertEqual(sum(x[4]-x[3] for x in got),140)
        self.assertTrue(sum(x[4]-x[3] for x in got)>=120)
        self.assertTrue(all(x[4]-x[3]<120 for x in got))

    def test_conflicting_labels_and_zero(self):
        seg = channel_segments([(0,100,0),(0,100,1),(40,60,2)])
        got = list(overlap_segments([(0,100)],seg))
        self.assertEqual(sum(x[4]-x[3] for x in got if x[2][2]==1),80)
        self.assertEqual(sum(x[4]-x[3] for x in got if x[2][5]=='ambiguous_labels'),20)
        self.assertEqual(list(overlap_segments([(0,100)],channel_segments([(0,100,0)])))[0][2][2],0)

    def test_channel_mismatch(self):
        schedules={1:channel_segments([(0,200,10)]),2:channel_segments([(0,200,20)])}
        got=list(overlap_segments([(0,200)],schedules[2]))
        self.assertEqual([x[2][2] for x in got],[20])
        self.assertNotIn(10,[x[2][2] for x in got])

    def test_adjacent_airings_stay_separate(self):
        self.assertEqual(len(channel_segments([(0,100,1),(100,200,1)])),2)
        self.assertEqual(list(union_intervals([(0,30),(30,50),(70,70)])),[(0,50)])

if __name__=='__main__': unittest.main()
