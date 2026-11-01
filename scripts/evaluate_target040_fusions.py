#!/usr/bin/env python3
"""Temporal evaluation of predeclared rank-only fusions from exact cached exports."""
import argparse,csv,gzip,json,math,time
from pathlib import Path
from collections import defaultdict
from fractions import Fraction
import pandas as pd
from audit_target040_round import exact_ap,digest,load
ROOT=Path(__file__).resolve().parents[1]

def run(config_path,out):
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);started=time.perf_counter();config=load(config_path);assert digest(config_path)==load(ROOT/config['freeze_evidence'])['sha256']
    lookup={s['name']:s for s in config['methods']};specs=[lookup[n] for n in config['portfolio_methods']]
    wanted={'ppr_v2','history_watches','rate_mix_seconds_50','eng_rate_recent_capped600','count_nmf16','count_nmf32','count_nmf64'}
    sources=[ROOT/('experiments/target040-round%02d-20261101-01'%n) for n in [5,6,7]];reference=load(sources[0]/'manifest.json')['input_hashes']
    assert all(load(p/'manifest.json')['input_hashes']==reference for p in sources)
    report={'config_sha256':digest(config_path),'source_input_hashes_identical':True,'folds':{},'overlapping_previously_observed_periods':True,'private_scores_used_in_metrics':False}
    for block in config['slices']:
        name=block['name'];directory=out/name;directory.mkdir();native={};input_hashes={}
        frozen=load(sources[0]/name/'cohort-and-targets.json');users=frozen['user_ids'];labels={r['user_id']:{'primary':set(r['primary_ordered_ids']),'secondary':set(r['secondary_ordered_ids']),'all_observed':set(r['all_observed_ids'])} for r in frozen['details']}
        for source in sources:
            assert load(source/name/'cohort-and-targets.json')['user_ids']==users
            path=source/name/'native-scores.csv.gz';input_hashes[str(path)]=digest(path);part=defaultdict(lambda:defaultdict(list))
            with gzip.open(path,'rt') as stream:
                for r in csv.DictReader(stream):
                    if r['component'] in wanted:part[r['user_id']][r['component']].append((int(r['item_id']),float(r['score'])))
            for user,components in part.items():
                target=native.setdefault(user,{})
                for component,records in components.items():
                    if component in target:
                        assert [i for i,_ in target[component]]==[i for i,_ in records]
                        assert all(abs(a-b)<1e-12 for (_,a),(_,b) in zip(target[component],records))
                    target[component]=records
        for user in users:
            for component in wanted:native.setdefault(user,{}).setdefault(component,[])
        prepared=ROOT/config['prepared'];pool=set(load(prepared/name/'candidates.json')['valid']);train=pd.read_parquet(prepared/name/'train-interactions.parquet');pop=train.groupby('item_id').user_id.nunique().to_dict();fallback=sorted(pool,key=lambda i:(-pop.get(i,0),i))
        segment_users=load(sources[2]/name/'segments.json')['users'];segments={u:s for s,us in segment_users.items() for u in us};acc=defaultdict(lambda:defaultdict(list));segment_values=defaultdict(lambda:defaultdict(lambda:defaultdict(list)));recall=defaultdict(list)
        with (directory/'per-user-predictions-and-AP.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=['method','user_id','ids','primary_AP','secondary_AP','all_observed_AP','primary_recall_native_union','secondary_recall_native_union','segment']);writer.writeheader()
            for spec in specs:
                for user in users:
                    if spec['kind']=='rank_fusion':
                        scores=defaultdict(float);union=set()
                        for component,weight in zip(spec['components'],spec['weights']):
                            for rank,(item,_) in enumerate(native[user][component],1):scores[item]+=weight/(60+rank);union.add(item)
                    else:scores=dict(native[user][spec['name']]);union=set(scores)
                    ids=[i for i,_ in sorted(scores.items(),key=lambda p:(-p[1],p[0]))][:5]
                    for item in fallback:
                        if len(ids)==5:break
                        if item not in ids:ids.append(item)
                    assert len(set(ids))==5 and set(ids)<=pool
                    row={'method':spec['name'],'user_id':user,'ids':' '.join(map(str,ids)),'segment':segments[user]}
                    for kind in ['primary','secondary','all_observed']:
                        value=exact_ap(ids,labels[user][kind]);acc[spec['name']][kind].append(value);segment_values[spec['name']][kind][segments[user]].append(value);row[kind+'_AP']=str(value)
                    for kind in ['primary','secondary']:row[kind+'_recall_native_union']=len(union&labels[user][kind])/len(labels[user][kind])
                    writer.writerow(row)
        values={n:{k:float(sum(v,Fraction(0))/len(v)) for k,v in kinds.items()} for n,kinds in acc.items()}
        per_segment={n:{k:{s:{'users':len(v),'MAP5':float(sum(v,Fraction(0))/len(v))} for s,v in segments_.items()} for k,segments_ in kinds.items()} for n,kinds in segment_values.items()}
        report['folds'][name]={'users':len(users),'candidates':len(pool),'metrics':values,'segments':per_segment,'source_native_hashes':input_hashes,'predictions_AP_sha256':digest(directory/'per-user-predictions-and-AP.csv')}
        print(json.dumps({'phase':'fold_complete','fold':name,'methods':len(specs),'users':len(users)}),flush=True)
    report['aggregate_dual_proxy']={n:sum((f['metrics'][n]['primary']+f['metrics'][n]['secondary'])/2 for f in report['folds'].values())/3 for n in config['portfolio_methods']};report['wall_seconds']=time.perf_counter()-started;report['status']='complete_rank_only_temporal_comparison';report['script_sha256']=digest(__file__)
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'phase':'complete','best':max(report['aggregate_dual_proxy'].items(),key=lambda p:p[1]),'seconds':report['wall_seconds']}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.config.resolve(),a.output.resolve())
