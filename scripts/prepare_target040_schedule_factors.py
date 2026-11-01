#!/usr/bin/env python3
"""Historical schedule descriptors for inductive implicit-MF projection.

No future views or content fields are read. Existing training vocabularies and
canonical same-item/conflict/deduplication rules are reused explicitly.
"""
import argparse
import datetime as dt
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy import sparse
from sweettv.data import sha256, timestamp, dump_json
from sweettv.v3_context import ContextState, schedule_features

ROOT = Path(__file__).resolve().parents[1]


def prepare(output, prepared, config_path, rate_root):
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    started = time.perf_counter()
    config = json.loads(config_path.read_text())
    base = ROOT/'experiments/competition-20261030-03'
    database = base/'raw.sqlite';before = sha256(database)
    audit = json.loads((base/'audit/data-audit.json').read_text())
    sources = [Path(__file__).resolve(),ROOT/'src/sweettv/v3_context.py',ROOT/'src/sweettv/data.py']
    source_hashes = {str(p.relative_to(ROOT)):sha256(p) for p in sources}
    inputs, outputs = {}, {}
    for block in config['slices']+[{'name':'final',**config['final']}]:
        name = block['name'];directory = prepared/name
        source = directory/'train-interactions.parquet'
        items = sorted(int(i) for i in pd.read_parquet(source,columns=['item_id']).item_id.unique())
        state_dir = prepared/'context'/('full' if name=='final' else name)
        state = ContextState.load(state_dir)
        lower,upper = timestamp(block['train_start']),timestamp(block['train_end'])
        assert (state.lower,state.cutoff)==(lower,upper)
        mean,ordered,totals,details = schedule_features(database,audit,state,items,lower,upper)
        assert ordered==items and np.isfinite(totals).all() and np.all(totals>0)
        historical = json.loads((rate_root/(name+'-train.json')).read_text())
        assert all(historical['seconds'].get(str(i),0.)==t for i,t in zip(items,totals))
        days = (upper-lower)/86400.
        matrix = (sparse.diags(totals/(3600.*days))@mean).tocsr()
        assert matrix.shape==(len(items),len(state.channels)*42)
        sparse.save_npz(output/(name+'-history.npz'),matrix)
        record = {'items':items,'lower':lower,'upper':upper,'days':days,'channels':state.channels,
                  'hours_per_day':(totals/(3600.*days)).tolist(),'feature_shape':list(matrix.shape),
                  'formula':'Canonical unique broadcast hours/day per training-channel x weekday x4h',
                  'future_viewing_used':False,'future_content_attributes_used':False,
                  'all_historical_totals_exactly_match_frozen_exposure':True,'audit':details}
        dump_json(output/(name+'-history.json'),record)
        for p in [source,directory/'availability.npz',directory/'availability.json',state_dir/'profile.json',rate_root/(name+'-train.json')]:
            inputs[str(p.resolve())]=sha256(p)
        print(json.dumps({'phase':'historical_schedule_descriptors','slice':name,'items':len(items),'nnz':matrix.nnz}),flush=True)
    for p in sorted(output.iterdir()):outputs[p.name]=sha256(p)
    assert before==sha256(database)
    assert source_hashes=={str(p.relative_to(ROOT)):sha256(p) for p in sources}
    assert all(sha256(Path(p))==h for p,h in inputs.items())
    manifest = {'status':'complete','created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
                'wall_seconds':time.perf_counter()-started,'database_before_and_after_sha256':before,
                'source_hashes':source_hashes,'input_hashes':inputs,'output_hashes':outputs,
                'source_config_sha256':sha256(config_path),'future_viewing_used':False,
                'future_content_attributes_used':False,'new_quality_metrics_computed':False}
    dump_json(output/'manifest.json',manifest)
    print(json.dumps({'status':'complete','wall_seconds':manifest['wall_seconds']}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--prepared',type=Path,default=ROOT/'experiments/target040-transfer-data-01')
    p.add_argument('--config',type=Path,default=ROOT/'config/target040-round05-preparation.json')
    p.add_argument('--rate-root',type=Path,default=ROOT/'experiments/target040-transfer-exposure-01')
    a=p.parse_args();prepare(a.output.resolve(),a.prepared.resolve(),a.config.resolve(),a.rate_root.resolve())
