#!/usr/bin/env python3
"""Refit a frozen, user-authorized verification portfolio with one shared fit."""
import argparse,copy,csv,datetime as dt,gzip,json,pickle,shutil,time
from pathlib import Path
from sweettv.data import sha256,dump_json
from sweettv.target040 import Family,read_slice,read_template,sources
from sweettv.target040_models import required_components
from sweettv.v3_models import SelectedModel
from sweettv.outputs import write_submission
ROOT=Path(__file__).resolve().parents[1]

def run(config_path,output):
    if output.exists():raise FileExistsError(output)
    config=json.loads(config_path.read_text());freeze=json.loads((ROOT/config['freeze_evidence']).read_text());assert sha256(config_path)==freeze['sha256']
    output.mkdir(parents=True);started=time.perf_counter()
    recorded=sources()
    for n in ['scripts/target040_portfolio.py','scripts/audit_target040_portfolio.py']:recorded[n]=sha256(ROOT/n)
    prepared=ROOT/config['prepared'];template=ROOT/'data/raw/extracted/submission.csv'
    _,train,recency,state,exposure,items,metadata=read_slice(prepared,config['final'],config,final=True)
    users=read_template(template,1260);lookup={s['name']:s for s in config['methods']};specs=[lookup[n] for n in config['portfolio_methods']]
    fit_specs=specs+[lookup[n] for n in ['rrf_ppr_nmf16_10','eng_rrf_recent_capped600']]
    input_hashes={str(p.relative_to(prepared)):sha256(p) for p in sorted(prepared.rglob('*')) if p.is_file()}
    manifest={'status':'running','started_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'source_hashes':recorded,'input_hashes':input_hashes,'prepared':str(prepared),'template':str(template),'expected_users':1260,'config_sha256':sha256(config_path),'selection_protocol':'Frozen local-evidence diversification portfolio, explicit latest user authorization','new_quality_metrics_computed':False}
    dump_json(output/'manifest.json',manifest);(output/'frozen-config.json').write_bytes(config_path.read_bytes())
    for n,h in recorded.items():
        p=output/'source-snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    if config.get('reuse_full_family_from'):
        previous=ROOT/config['reuse_full_family_from'];old=json.loads((previous/'manifest.json').read_text())
        assert old['status']=='complete_target040_portfolio' and old['input_hashes']==input_hashes
        with (previous/'fitted-components.pkl').open('rb') as stream:family=pickle.load(stream)
        needed=required_components(fit_specs,config['methods'])
        family.models={n:m for n,m in family.models.items() if n in needed}
        family.config=config
        manifest['reused_full_family_sha256']=sha256(previous/'fitted-components.pkl')
        manifest['reused_full_family_source_manifest_sha256']=sha256(previous/'manifest.json')
    else:
        family=Family(train,metadata['train'],recency,state,exposure,items,config,fit_specs)
    if family.failures:raise RuntimeError('Portfolio component unavailable: '+str(family.failures))
    print(json.dumps({'phase':'shared_fit_complete','seconds':time.perf_counter()-started,'components':len(family.models)}),flush=True)
    with (output/'fitted-components.pkl').open('wb') as stream:pickle.dump(family,stream,pickle.HIGHEST_PROTOCOL)
    native={};predictions={n:{} for n in config['portfolio_methods']};controls={n:{} for n in ['rrf_ppr_nmf16_10','eng_rrf_recent_capped600']}
    for index,user in enumerate(users):
        lists=family.lists(user);native[user]=lists
        for spec in specs:
            records,_,_=family.recommend_from(user,spec,lists);predictions[spec['name']][user]=[r['item_id'] for r in records]
        for name in controls:
            records,_,_=family.recommend_from(user,lookup[name],lists);controls[name][user]=[r['item_id'] for r in records]
        if index%200==0:print(json.dumps({'phase':'inference','users':index}),flush=True)
    with gzip.open(output/'native-lists.json.gz','wt') as stream:json.dump(native,stream)
    for name,expected in [('rrf_ppr_nmf16_10','072d7c50e5d9b3d6da83c4f0fa0ae956382ed23c2898daeafbc24333e66cc01d'),('eng_rrf_recent_capped600','8cefdc8efeeeebd2d6a97959d37f32afe4e222ba2701ef70b2f8aea917d63789')]:
        p=output/'controls'/name/'submission.csv';p.parent.mkdir(parents=True,exist_ok=True);check=write_submission(template,p,controls[name],set(items),1260);assert check['sha256']==expected
    dump_json(output/'control-reproduction.json',{'passed':True,'exact_previous_full_CSV_controls':list(controls),'users_per_control':1260})
    for spec in specs:
        name=spec['name'];directory=output/'candidates'/name;final=directory/'final';final.mkdir(parents=True)
        check=write_submission(template,final/'submission.csv',predictions[name],set(items),1260)
        subset=copy.copy(family);needed=required_components([spec],config['methods']);subset.models={n:m for n,m in family.models.items() if n in needed}
        selected=SelectedModel(subset,spec);selected.save(final/'model.pkl')
        with (final/'model.pkl').open('rb') as stream:restored=pickle.load(stream)
        for user in list(dict.fromkeys(users[:3]+config['examples'])):assert [r['item_id'] for r in restored.recommend(user)]==predictions[name][user]
        dump_json(final/'submission-validation.json',{**check,'selected_specification':spec,'frozen_portfolio':str(output),'source_local_protocols':config['portfolio_provenance'][name],'model_roundtrip_checked':True})
        dump_json(directory/'selection.json',{'selected_method':name,'selected_specification':spec,'selection_type':'predeclared verification portfolio','selection_rule':config['selection'],'source_local_protocols':config['portfolio_provenance'][name],'kaggle_scores_used':False})
        (directory/'frozen-config.json').write_bytes(config_path.read_bytes())
        child={**manifest,'status':'complete_local_target040_portfolio_candidate','parent_portfolio':str(output),'source_snapshot_root':str(output/'source-snapshot'),'final_model_sha256':sha256(final/'model.pkl'),'final_csv_sha256':check['sha256']}
        dump_json(directory/'manifest.json',child)
        print(json.dumps({'phase':'candidate_saved','method':name,'sha256':check['sha256']}),flush=True)
    for n,h in recorded.items():assert sha256(ROOT/n)==h
    for n,h in input_hashes.items():assert sha256(prepared/n)==h
    manifest.update(status='complete_target040_portfolio',wall_seconds=time.perf_counter()-started,finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(),controls_reproduced_exactly=True)
    dump_json(output/'manifest.json',manifest)
    print(json.dumps({'phase':'complete','wall_seconds':manifest['wall_seconds'],'candidates':len(specs)}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=ROOT/'config/target040-portfolio01.json');p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.config.resolve(),a.output.resolve())
