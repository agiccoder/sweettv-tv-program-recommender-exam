#!/usr/bin/env python3
"""Freshly refit one preserved candidate and require exact CSV reproduction.

Feature files can be rebuilt at new paths. Their actual hashes are frozen before
fitting; model settings, windows, candidate policy and chosen specification stay
those of the original scored candidate. This does not run a parameter search.
"""
import argparse,copy,datetime as dt,json,pickle,time
from pathlib import Path
from sweettv.data import sha256,dump_json
from sweettv.target040 import Family,read_slice,read_template,sources
from sweettv.v3_models import SelectedModel
from sweettv.outputs import write_submission
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
def run(a):
    if a.output.exists():raise FileExistsError(a.output)
    original=read(a.source_run/'frozen-config.json');manifest=read(a.source_run/'manifest.json');selection=read(a.source_run/'selection.json')
    assert sha256(a.source_run/'frozen-config.json')==manifest['config_sha256']
    config=copy.deepcopy(original);spec=selection['selected_specification'];config['reuse_cached_fits']=False
    prepared=(a.prepared or ROOT/config['prepared']).resolve();config['prepared']=str(prepared)
    if a.rate_features:config['rate_features_root']=str(a.rate_features.resolve())
    if a.engagement_features:config['engagement']['root']=str(a.engagement_features.resolve())
    if a.schedule_features:config['inductive_mf']['root']=str(a.schedule_features.resolve())
    features={}
    for key,hash_key in [('rate_features_root','rate_features_hashes')]:
        if key in config:
            directory=ROOT/config[key];actual={p.name:sha256(p) for p in directory.iterdir() if p.is_file()}
            config[hash_key]=actual;features[str(directory)]=actual
    for key in ['engagement','inductive_mf']:
        if key in config:
            directory=ROOT/config[key]['root'];actual={p.name:sha256(p) for p in directory.iterdir() if p.is_file()}
            config[key]['hashes']=actual;features[str(directory)]=actual
    template=(a.template or ROOT/'data/raw/extracted/submission.csv').resolve();users=read_template(template,1260)
    a.output.mkdir(parents=True);started=time.perf_counter()
    frozen={'created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'original_config_sha256':manifest['config_sha256'],
        'selected_specification':spec,'fitting_config':config,'expected_csv_sha256':a.expected_sha256,
        'prepared_input_hashes':{str(p.relative_to(prepared)):sha256(p) for p in prepared.rglob('*') if p.is_file()},
        'feature_hashes':features,'source_hashes':sources(),'script_sha256':sha256(__file__),
        'new_quality_metrics_or_search':False,'feature_manifest_hashes_refrozen_before_fit':True}
    dump_json(a.output/'refit-freeze.json',frozen)
    _,train,recency,state,exposure,items,metadata=read_slice(prepared,config['final'],config,final=True)
    family=Family(train,metadata['train'],recency,state,exposure,items,config,[spec])
    assert not family.failures
    selected=SelectedModel(family,spec);predictions={u:[r['item_id'] for r in selected.recommend(u)] for u in users}
    check=write_submission(template,a.output/'submission.csv',predictions,set(items),1260)
    assert check['sha256']==a.expected_sha256,'Reproduction differs; preserve actual evidence and diagnose'
    selected.save(a.output/'model.pkl')
    with (a.output/'model.pkl').open('rb') as stream:restored=pickle.load(stream)
    for u in list(dict.fromkeys(users[:3]+config['examples'])):assert [r['item_id'] for r in restored.recommend(u)]==predictions[u]
    assert sources()==frozen['source_hashes']
    dump_json(a.output/'reproduction-validation.json',{**check,'exact_expected_CSV_bytes_reproduced':True,
        'fresh_refit_no_fitted_cache':True,'model_roundtrip_passed':True,'wall_seconds':time.perf_counter()-started,
        'refit_freeze_sha256':sha256(a.output/'refit-freeze.json'),'model_sha256':sha256(a.output/'model.pkl')})
    print(json.dumps({'passed':True,'method':spec['name'],'CSV_sha256':check['sha256'],'fresh_refit':True}))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--expected-sha256',required=True);p.add_argument('--prepared',type=Path);p.add_argument('--template',type=Path);p.add_argument('--rate-features',type=Path);p.add_argument('--engagement-features',type=Path);p.add_argument('--schedule-features',type=Path);run(p.parse_args())
