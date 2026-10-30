"""Hand-computable cutoff/airing/decay fixtures; no competition score claims."""
import datetime as dt
from pathlib import Path
import sqlite3
import tempfile
import unittest

import pandas as pd

from sweettv.time_features import COLUMNS, build_recency_features

BASE=dt.datetime(2020,1,1,tzinfo=dt.timezone.utc).timestamp()
DAY=86400
USER='000123456789012345678901234567890'


class TimeFeatureTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.db=self.root/'raw.sqlite'
        self.conn=sqlite3.connect(self.db)
        self.conn.executescript('CREATE TABLE views(user_id TEXT,channel INTEGER,start REAL,end REAL,duration REAL);'
                                'CREATE TABLE schedule(role TEXT,channel INTEGER,start REAL,duration REAL,item_id INTEGER);')
        self.audit={'files':{r:{'duration_scale_seconds':1,'sha256':None} for r in ['train','valid']}}

    def airing(self,item,start,end,channel=1,role='train'):
        self.conn.execute('INSERT INTO schedule VALUES(?,?,?,?,?)',(role,channel,BASE+start,end-start,item))

    def view(self,start,end,channel=1,user=USER):
        self.conn.execute('INSERT INTO views VALUES(?,?,?,?,?)',(user,channel,BASE+start,BASE+end,end-start))

    def run_features(self,windows,**kw):
        self.conn.commit();self.conn.close()
        out=self.root/'features'
        audit=build_recency_features(self.db,out,self.audit,windows={k:(BASE+a,BASE+b) for k,(a,b) in windows.items()},**kw)
        frames={name:pd.read_parquet(out/(name+'.parquet')) for name in windows}
        return audit,frames

    def test_fragment_union_threshold_and_one_half_life_by_hand(self):
        self.airing(11,0,200);self.airing(11,0,200);self.airing(11,30,150)
        self.view(0,70);self.view(0,70);self.view(100,170)
        self.view(0,200,channel=999) # Channel mismatch cannot create an item.
        audit,frames=self.run_features({'early':(0,170+28*DAY)})
        frame=frames['early'];self.assertEqual(list(frame),COLUMNS);self.assertEqual(len(frame),1)
        row=frame.iloc[0];self.assertEqual(row.user_id,USER)
        self.assertEqual(row.seconds,140);self.assertEqual(row.watches,1)
        self.assertEqual(row.last_event_end,BASE+170)
        self.assertAlmostEqual(row.recent_seconds,70,places=12)
        self.assertAlmostEqual(row.recent_watches,.5,places=12)
        self.assertEqual(audit['windows']['early']['airings']['accepted_user_airings'],1)

    def test_threshold_is_per_canonical_airing_not_item_and_boundaries_are_half_open(self):
        self.airing(11,0,75);self.airing(11,75,150);self.airing(11,300,450)
        self.view(0,150);self.view(300,450);self.view(450,600)
        _,frames=self.run_features({'early':(0,450)})
        row=frames['early'].iloc[0]
        self.assertEqual(row.seconds,150);self.assertEqual(row.watches,1)
        self.assertEqual(row.last_event_end,BASE+450)
        self.assertAlmostEqual(row.recent_seconds,150)

    def test_cutoff_clips_before_threshold_and_later_views_do_not_change_early_profile(self):
        self.airing(11,0,300);self.airing(22,1000,1300,role='valid')
        self.view(0,300);self.view(1000,1300)
        _,frames=self.run_features({'short':(0,90),'early':(0,180),'later':(0,1300)})
        self.assertTrue(frames['short'].empty)
        early=frames['early'].iloc[0]
        self.assertEqual(early.seconds,180);self.assertEqual(early.last_event_end,BASE+180)
        self.assertEqual(set(frames['early'].item_id),{11})
        self.assertEqual(set(frames['later'].item_id),{11,22})
        self.assertTrue((frames['early'].last_event_end<=BASE+180).all())

    def test_conflicts_zero_ids_and_future_schedule_are_not_preferences(self):
        self.airing(11,0,300);self.airing(22,100,200)
        self.airing(0,0,300);self.airing(33,400,700,role='future')
        self.view(0,300);self.view(400,700)
        _,frames=self.run_features({'early':(0,1000)})
        self.assertEqual(set(frames['early'].item_id),{11})
        row=frames['early'].iloc[0];self.assertEqual(row.seconds,200);self.assertEqual(row.watches,1)
        self.assertEqual(row.last_event_end,BASE+300)

    def test_exact_reference_parity_is_consumed_and_mismatch_fails(self):
        self.airing(11,0,300);self.view(0,300)
        ref=self.root/'original.parquet'
        pd.DataFrame([{'user_id':USER,'item_id':11,'seconds':300.,'watches':1}]).to_parquet(ref,index=False)
        audit,frames=self.run_features({'early':(0,300)},references={'early':ref})
        self.assertTrue(audit['windows']['early']['exact_original_interaction_parity']['passed'])
        from sweettv.time_features import _check_reference
        altered=frames['early'].copy();altered.loc[0,'seconds']=301
        with self.assertRaises(AssertionError):_check_reference(altered,ref)
        with self.assertRaises(FileExistsError):
            build_recency_features(self.db,self.root/'features',self.audit,windows={'early':(BASE,BASE+300)})

    def test_decay_is_summed_per_accepted_airing_not_latest_item_timestamp(self):
        self.airing(11,0,120);self.airing(11,28*DAY,28*DAY+120)
        self.view(0,120);self.view(28*DAY,28*DAY+120)
        _,frames=self.run_features({'early':(0,28*DAY+120)})
        row=frames['early'].iloc[0]
        self.assertEqual(row.seconds,240);self.assertEqual(row.watches,2)
        self.assertAlmostEqual(row.recent_seconds,180,places=12)
        self.assertAlmostEqual(row.recent_watches,1.5,places=12)


if __name__=='__main__':unittest.main()
