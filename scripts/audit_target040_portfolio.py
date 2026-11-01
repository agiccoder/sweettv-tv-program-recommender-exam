#!/usr/bin/env python3
"""Independent all-user portfolio composition/CSV/provenance verification."""
import argparse,csv,gzip,json,math
from pathlib import Path
from collections import defaultdict
import pandas as pd
from audit_target040_round import component_spec,history_rank,digest,load
ROOT=Path(__file__).resolve().parents[1]

def audit(root):
    manifest=load(root/'manifest.json');config=load(root/'frozen-config.json');assert manifest['status']=='complete_target040_portfolio'
    assert load(root/'control-reproduction.json')['passed']
    freeze=load(ROOT/config['freeze_evidence']);assert digest(root/'frozen-config.json')==freeze['sha256']==manifest['config_sha256']
    for name,value in manifest['source_hashes'].items():assert digest(root/'source-snapshot'/name)==value
    prepared=Path(manifest['prepared']);pool=set(load(prepared/'final/candidates.json')['valid'])
    template=list(csv.reader(Path(manifest['template']).open(encoding='utf-8-sig')));users=[r[0] for r in template[1:]]
    train=pd.read_parquet(prepared/'final/train-interactions.parquet');viewers=train.groupby('item_id').user_id.nunique().to_dict();fallback=sorted(pool,key=lambda item:(-viewers.get(item,0),item))
    with gzip.open(root/'native-lists.json.gz','rt') as stream:native=json.load(stream)
    history={u:history_rank(native,u,config['fusion']['rrf_constant']) for u in users} if 'history_seconds' in next(iter(native.values())) else {}
    lookup={s['name']:s for s in config['methods']};results=[]
    for name in config['portfolio_methods']:
        directory=root/'candidates'/name;spec=lookup[name];check=load(directory/'final/submission-validation.json');assert check['passed']
        actual=list(csv.reader((directory/'final/submission.csv').open(encoding='utf-8-sig')))
        assert actual[0]==template[0] and len(actual)==1261 and [r[0] for r in actual[1:]]==users
        for user,text in actual[1:]:
            tokens=text.split(' ');ids=list(map(int,tokens));assert len(ids)==5 and len(set(ids))==5 and min(ids)>0 and set(ids)<=pool and ' '.join(map(str,ids))==text
            components=component_spec(spec,native,history,user)
            if components is None:scores={int(i):float(v) for i,v in native[user][name]}
            else:
                scores=defaultdict(float)
                for _,(ranked,weight) in components.items():
                    for item,(rank,_) in ranked.items():scores[item]+=weight/(config['fusion']['rrf_constant']+rank)
            ordered=sorted(((i,v) for i,v in scores.items() if i in pool and math.isfinite(v) and v>0),key=lambda x:(-x[1],x[0]))
            expected=[i for i,_ in ordered][:5]
            for item in fallback:
                if len(expected)>=5:break
                if item not in expected:expected.append(item)
            assert ids==expected,(name,user)
        sha=digest(directory/'final/submission.csv');assert sha==check['sha256']
        report={'passed':True,'performed_by':'Codex independent portfolio composition/CSV audit','script_sha256':digest(__file__),'final_csv_sha256':sha,'rows_checked':1260,'recommendation_slots_checked':6300,'full_native_ranking_and_training_popularity_fallback_reproduced':True,'exact_template_strings_order_headers_and_ids':True,'source_snapshot_verified':True,'prior_full_CSV_controls_reproduced':True,'model_roundtrip_passed':check['model_roundtrip_checked']}
        path=directory/'independent-audit.json';assert not path.exists();path.write_text(json.dumps(report,indent=2)+'\n');results.append({'method':name,'csv_sha256':sha,'passed':True})
    path=root/'portfolio-independent-audit.json';assert not path.exists();path.write_text(json.dumps({'passed':True,'candidates':results,'total_slots_checked':len(results)*6300,'script_sha256':digest(__file__)},indent=2)+'\n');print(json.dumps({'passed':True,'candidates':len(results),'slots':len(results)*6300}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();audit(a.root.resolve())
