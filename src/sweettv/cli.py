"""Reproducible CLI. Competition results require genuine authorized inputs."""
from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import traceback

import pandas as pd

from .data import build_interactions, discover, dump_json, extract_archive, import_data, rows, sha256
from .metrics import audit_exported_ap, evaluate
from .models import fit_content, fit_popularity, fit_ppr
from .outputs import validate_submission, write_submission

SEED = 20261030
SEARCH = {'threshold_seconds': 120, 'repeat_policy': 'allow', 'ppr_alphas': [.5,.85],
          'ppr_max_neighbors': 100, 'ppr_tol': 1e-8, 'ppr_max_iter': 300,
          'content': 'binary metadata tokens, smooth IDF, L2 vectors, log1p(seconds) profile',
          'selection': 'highest common-cohort MAP@5; exact ties popularity, content, ppr_0.5, ppr_0.85',
          'seed': SEED, 'hybrid': None}


def environment():
    distributions = ['numpy','pandas','scipy','scikit-learn','pyarrow','reportlab','PyMuPDF','requests','kaggle','playwright']
    versions = {}
    for name in distributions:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    cpu = subprocess.run(['sysctl','-n','machdep.cpu.brand_string'],capture_output=True,text=True).stdout.strip() if sys.platform=='darwin' else platform.processor()
    memory = subprocess.run(['sysctl','-n','hw.memsize'],capture_output=True,text=True).stdout.strip() if sys.platform=='darwin' else None
    return {'python': sys.version, 'platform': platform.platform(), 'machine': platform.machine(),
            'cpu': cpu, 'physical_memory_bytes': int(memory) if memory else None,
            'packages': versions, 'threads': {k: os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']},
            'sqlite': __import__('sqlite3').sqlite_version}


def source_hashes():
    root = Path(__file__).resolve().parents[2]
    # Report and native-export scripts do not fit predictions; their own output
    # manifests record their hashes independently of a model run.
    paths = list((root/'src').rglob('*.py')) + list((root/'tests').rglob('*.py'))
    for name in ['requirements.lock','pyproject.toml']:
        if (root/name).exists():
            paths.append(root/name)
    paths += list((root/'config').rglob('*.json'))
    return {str(p.relative_to(root)): sha256(p) for p in sorted(paths)}


def load_json(path):
    return json.loads(Path(path).read_text())


def load_meta(path):
    return {w: {int(i):v for i,v in items.items()} for w,items in load_json(path).items()}


def write_csv(path, records, fields):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def evaluate_run(run_dir, template_path, expected_users=1260):
    run_dir = Path(run_dir)
    prepared = run_dir/'prepared'
    train = pd.read_parquet(prepared/'train-interactions.parquet')
    valid = pd.read_parquet(prepared/'valid-interactions.parquet')
    metadata = load_meta(prepared/'metadata.json')
    candidates = load_json(prepared/'candidates.json')['valid']
    pool=set(candidates)
    template_records=list(rows(template_path))
    template_users=[r['user_id'] for r in template_records]
    if len(template_users)!=expected_users or len(set(template_users))!=expected_users:
        raise ValueError('Template cohort size or uniqueness mismatch')
    raw_targets=collections.defaultdict(set)
    for row in valid.itertuples():
        raw_targets[row.user_id].add(int(row.item_id))
    targets={u:raw_targets[u]&pool for u in template_users}
    cohort=[u for u in template_users if targets[u]]
    if len(cohort)<2:
        raise ValueError('Fewer than two template users have observable eligible holdout labels')
    training_counts=train.groupby('user_id').item_id.nunique().to_dict()
    warm=set(training_counts)
    reasons=[]
    for u in template_users:
        reasons.append({'user_id':u,'in_cohort':bool(targets[u]),
                        'no_observable_holdout_interactions':not raw_targets[u],
                        'no_eligible_holdout_targets':not targets[u],
                        'some_targets_outside_candidates':bool(raw_targets[u]-pool),
                        'outside_target_count':len(raw_targets[u]-pool),
                        'no_usable_training_history':u not in warm,
                        'relevant_ids':sorted(targets[u])})
    cohort_audit={'evaluation_users':len(cohort),'template_users':len(template_users),
                  'eligible_zero_history_users':sum(u not in warm for u in cohort),
                  'excluded_no_eligible_targets':len(template_users)-len(cohort),
                  'holdout_users_outside_template':len(set(raw_targets)-set(template_users)),
                  'reason_counts_overlapping': {k:sum(bool(r[k]) for r in reasons) for k in [
                      'no_observable_holdout_interactions','no_eligible_holdout_targets',
                      'some_targets_outside_candidates','no_usable_training_history']},
                  'empty_target_convention':'AP=0 in metric; users with no observable eligible target excluded from primary cohort before all methods',
                  'user_ids':cohort,'details':reasons}
    dump_json(run_dir/'cohort.json',cohort_audit)
    # Fixed examples selected BEFORE any model scores or AP are computed.
    heavy=sorted(cohort,key=lambda u:(-training_counts.get(u,0),u))[0]
    light=sorted((u for u in cohort if u!=heavy),key=lambda u:(training_counts.get(u,0),u))[0]
    examples={'selection':'maximum/minimum distinct usable training items among fixed cohort; ties lexicographic raw ID; light excludes heavy',
              'users':[{'role':'heavy','user_id':heavy,'training_items':training_counts.get(heavy,0)},
                       {'role':'light','user_id':light,'training_items':training_counts.get(light,0)}]}
    dump_json(run_dir/'example-users.json',examples)
    output, all_ap, rec_records, trained, timings = {}, [], [], {}, {}
    factories=[('popularity',lambda:fit_popularity(train)),
               ('content',lambda:fit_content(train,metadata['train']))]
    factories += [('ppr_'+str(alpha),lambda a=alpha:fit_ppr(train,alpha=a)) for alpha in SEARCH['ppr_alphas']]
    for name,factory in factories:
        start=time.perf_counter()
        model=factory()
        fit_seconds=time.perf_counter()-start
        start=time.perf_counter()
        recommendations={u:model.recommend(u,candidates) for u in cohort}
        predict_seconds=time.perf_counter()-start
        ids={u:[r['item_id'] for r in recs] for u,recs in recommendations.items()}
        for user,items in ids.items():
            if len(items)!=5 or len(set(items))!=5 or not set(items)<=pool:
                raise AssertionError(f'Invalid validation Top5 {name} {user}')
        result=evaluate(ids,targets,cohort)
        ap=result.pop('per_user_ap')
        warm_n=sum(u in warm for u in cohort)
        cold_n=len(cohort)-warm_n
        result.update({'fit_seconds':fit_seconds,'predict_seconds':predict_seconds,
                       'candidate_count':len(candidates),'warm_users':warm_n,'cold_users':cold_n,
                       'warm_map_at_5':sum(ap[u] for u in cohort if u in warm)/warm_n if warm_n else None,
                       'cold_map_at_5':sum(ap[u] for u in cohort if u not in warm)/cold_n if cold_n else None})
        output[name]=result
        timings[name]={'fit_seconds':fit_seconds,'predict_seconds':predict_seconds}
        for u in cohort:
            all_ap.append({'method':name,'user_id':u,'relevant_ids':json.dumps(sorted(targets[u])),
                           'predictions':json.dumps(ids[u]),'ap_at_5':ap[u]})
            for rank,r in enumerate(recommendations[u],1):
                meta=metadata['valid'].get(r['item_id'],{})
                rec_records.append({'method':name,'user_id':u,'rank':rank,**r,
                                    'title':meta.get('tv_show_title',''),'category':meta.get('tv_show_category',''),
                                    'relevant':r['item_id'] in targets[u]})
        model.save(run_dir/'models'/f'{name}.pkl')
        dump_json(run_dir/'diagnostics'/f'{name}.json',model.diagnostics)
        trained[name]=model
    ordered=[name for name,_ in factories]
    selected=max(ordered,key=lambda n:output[n]['map_at_5'])
    selected_ppr=max([n for n in ordered if n.startswith('ppr_')],key=lambda n:output[n]['map_at_5'])
    summary={'status':'complete','methods':output,'selected_method':selected,'selected_ppr':selected_ppr,
             'search':SEARCH,'cohort_size':len(cohort),'selection_source':'local temporal validation only',
             'competition_metric_equivalence':'not established; slides convention implemented locally'}
    dump_json(run_dir/'validation-summary.json',summary)
    write_csv(run_dir/'per-user-ap.csv',all_ap,['method','user_id','relevant_ids','predictions','ap_at_5'])
    write_csv(run_dir/'validation-recommendations.csv',rec_records,['method','user_id','rank','item_id','score','reason','title','category','relevant'])
    audit=audit_exported_ap(run_dir/'per-user-ap.csv',expected_map={n:r['map_at_5'] for n,r in output.items()})
    dump_json(run_dir/'independent-ap-audit.json',audit)
    demo_methods=['popularity','content',selected_ppr]
    history=[]
    for w,frame in [('train',train),('valid',valid)]:
        for row in frame[frame.user_id.isin([heavy,light])].itertuples():
            meta=metadata[w].get(int(row.item_id),{})
            history.append({'window':w,'user_id':row.user_id,'item_id':int(row.item_id),
                            'seconds':float(row.seconds),'watches':int(row.watches),
                            'title':meta.get('tv_show_title',''),'category':meta.get('tv_show_category','')})
    demo=[r for r in rec_records if r['user_id'] in [heavy,light] and r['method'] in demo_methods]
    dump_json(run_dir/'two-user-comparison.json',{'selection':examples,'methods':demo_methods,
                                                'histories':history,'recommendations':demo,
                                                'period':'validation only; no hidden future labels'})
    return summary


def final_predict(run_dir,template_path,expected_users=1260):
    run_dir=Path(run_dir)
    prepared=run_dir/'prepared'
    summary=load_json(run_dir/'validation-summary.json')
    selected=summary['selected_method']
    full=pd.read_parquet(prepared/'full-interactions.parquet')
    metadata=load_meta(prepared/'metadata.json')
    full_metadata=dict(metadata['train'])
    for item,record in metadata['valid'].items():
        prior=full_metadata.setdefault(item,{})
        for key,value in record.items():
            if not prior.get(key) and value:
                prior[key]=value
    candidates=load_json(prepared/'candidates.json')['future']
    start=time.perf_counter()
    if selected=='popularity':
        model=fit_popularity(full)
    elif selected=='content':
        model=fit_content(full,full_metadata)
    else:
        model=fit_ppr(full,alpha=float(selected.split('_')[1]))
    fit_seconds=time.perf_counter()-start
    start=time.perf_counter()
    users=[r['user_id'] for r in rows(template_path)]
    recs={u:model.recommend(u,candidates) for u in users}
    predict_seconds=time.perf_counter()-start
    ids={u:[r['item_id'] for r in values] for u,values in recs.items()}
    out=run_dir/'final'
    check=write_submission(template_path,out/'submission.csv',ids,set(candidates),expected_users)
    check.update({'selected_method':selected,'fit_seconds':fit_seconds,'predict_seconds':predict_seconds,
                  'fit_window':'full historical weeks 11-30, reconstructed without validation cutoff threshold splitting',
                  'future_preference_features':False})
    dump_json(out/'submission-validation.json',check)
    model.save(out/'model.pkl')
    dump_json(out/'model-diagnostics.json',model.diagnostics)
    rec_records=[]
    for user,values in recs.items():
        for rank,row in enumerate(values,1):
            meta=metadata['future'].get(row['item_id'],{})
            rec_records.append({'user_id':user,'rank':rank,**row,'title':meta.get('tv_show_title',''),
                                'category':meta.get('tv_show_category','')})
    write_csv(out/'recommendations.csv',rec_records,['user_id','rank','item_id','score','reason','title','category'])
    return check


def run(args):
    target=Path(args.run_dir)
    if target.exists():
        raise FileExistsError('Immutable run directory already exists; select a new --run-dir')
    target.mkdir(parents=True)
    start=time.perf_counter()
    manifest={'started_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'status':'running',
              'search_frozen_before_evaluation':SEARCH,'environment':environment(),
              'calendar_boundaries':getattr(args,'calendar_boundaries',None),
              'source_hashes':source_hashes(),'synthetic_fixture':bool(args.synthetic_fixture),
              'external_uploads':False}
    dump_json(target/'manifest.json',manifest)
    try:
        files=discover(args.raw_dir)
        audit=import_data(args.raw_dir,target/'raw.sqlite',target/'audit',args.duration_units)
        build_interactions(target/'raw.sqlite',audit,target/'prepared',SEARCH['threshold_seconds'],getattr(args,'calendar_boundaries',None))
        evaluate_run(target,files['template'],args.expected_users)
        final_predict(target,files['template'],args.expected_users)
        if source_hashes()!=manifest['source_hashes']:
            raise RuntimeError('Source changed during the run; results are not a reproducible final run')
        manifest['status']='complete_synthetic_fixture' if args.synthetic_fixture else 'complete_local_competition_run'
    except Exception as exc:
        manifest['status']='failed'
        dump_json(target/'failure.json',{'exception_type':type(exc).__name__,'message':str(exc),
                                        'traceback':traceback.format_exc(),'competition_results_produced':False,
                                        'synthetic_fixture':bool(args.synthetic_fixture)})
        raise
    finally:
        manifest['finished_utc']=dt.datetime.now(dt.timezone.utc).isoformat()
        manifest['wall_seconds']=time.perf_counter()-start
        memory=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        manifest['peak_resident_memory_bytes']=memory if sys.platform=='darwin' else memory*1024
        manifest['source_hashes_at_end']=source_hashes()
        manifest['source_hashes_match']=manifest['source_hashes']==manifest['source_hashes_at_end']
        dump_json(target/'manifest.json',manifest)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    subs=parser.add_subparsers(dest='command',required=True)
    p=subs.add_parser('run')
    p.add_argument('--raw-dir',default='data/raw')
    p.add_argument('--run-dir',required=True)
    p.add_argument('--expected-users',type=int,default=1260)
    p.add_argument('--synthetic-fixture',action='store_true')
    p.add_argument('--duration-units',type=json.loads,default=None,help='JSON role -> explicit seconds multiplier; only after raw audit')
    p.add_argument('--calendar-boundaries',type=json.loads,default=None,help='Explicit observed window-start dates, plus optional end; required for schedule spillovers')
    p.add_argument('--calendar-config',type=Path,default=None,help='JSON file containing explicit observed calendar boundaries')
    p=subs.add_parser('extract')
    p.add_argument('archive')
    p.add_argument('--destination',default='data/raw/extracted')
    p=subs.add_parser('validate')
    p.add_argument('--run-dir',required=True)
    p.add_argument('--template',required=True)
    p.add_argument('--expected-users',type=int,default=1260)
    args=parser.parse_args()
    if args.command=='run':
        if args.calendar_config:
            if args.calendar_boundaries:
                parser.error('Use only one calendar specification')
            args.calendar_boundaries=load_json(args.calendar_config)
        run(args)
    elif args.command=='extract':
        print(json.dumps(extract_archive(args.archive,args.destination),indent=2))
    else:
        target=Path(args.run_dir)
        audit_exported_ap(target/'per-user-ap.csv',target/'validation-summary.json')
        print(json.dumps(validate_submission(args.template,target/'final'/'submission.csv',
                                           set(load_json(target/'prepared'/'candidates.json')['future']),args.expected_users),indent=2))


if __name__=='__main__':
    main()
