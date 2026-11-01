#!/usr/bin/env python3
"""Reconcile accepted hashes and resume serially after a server rate-limit pause."""
import argparse,datetime as dt,json,re,time,requests
from pathlib import Path
from target040_kaggle import validate_run,digest,read,submission_record,COMPETITION
ROOT=Path(__file__).resolve().parents[1]

def main(root,out):
    if out.exists():raise FileExistsError(out)
    assert read(root/'portfolio-independent-audit.json')['passed'];out.mkdir(parents=True)
    limited=read(ROOT/'evidence/target040-01/portfolio01-submissions/rate-limit-observation.json')
    elapsed=(dt.datetime.now(dt.timezone.utc)-dt.datetime.fromisoformat(limited['observed_utc'])).total_seconds()
    if elapsed<120:
        print(json.dumps({'phase':'respect_server_cooldown','seconds_remaining':120-elapsed}),flush=True)
        time.sleep(120-elapsed)
    from kaggle import api
    username=api.get_config_value('username');key=api.get_config_value('key')
    quota=requests.get('https://www.kaggle.com/api/v1/competitions/'+COMPETITION+'/submissions/limits',auth=(username,key),timeout=30);quota.raise_for_status();limits=quota.json()
    (out/'live-limits-before.json').write_text(json.dumps(limits,indent=2)+'\n')
    observations=[]
    for page in range((limits['numTotal']+19)//20):
        batch=api.competition_submissions(COMPETITION,page_token=page);observations.extend(submission_record(x) for x in batch)
        if page+1<(limits['numTotal']+19)//20:time.sleep(3)
    (out/'submissions-before.json').write_text(json.dumps(observations,indent=2)+'\n')
    known={}
    for p in (ROOT/'evidence/target040-01').rglob('upload-receipt.json'):
        x=read(p);known[x['candidate_sha256']]=x['submission_id']
    for x in read(ROOT/'evidence/target040-01/historical-submissions/linkage-register.json'):known[x['downloaded_sha256']]=x['submission_id']
    for x in observations:
        match=re.search(r'sha256 ([0-9a-f]{64})',x.get('description') or '')
        if match:known[match.group(1)]=x['id']
    queue=[];skipped=[]
    for x in read(root/'portfolio-independent-audit.json')['candidates']:
        run=root/'candidates'/x['method'];source,sha,selection=validate_run(run)
        if sha in known:skipped.append({'method':x['method'],'sha256':sha,'existing_id':known[sha]});continue
        known[sha]='queued';queue.append((x['method'],run,source,sha,selection))
    assert len(queue)<=limits['numAllowedNow']
    (out/'frozen-upload-queue.json').write_text(json.dumps({'created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'methods':[x[0] for x in queue],'skipped':skipped,'pacing_seconds':5,'script_sha256':digest(__file__),'no_unknown_outcome_retried_before_hash_reconciliation':True},indent=2)+'\n')
    uploaded=[]
    for name,run,source,sha,selection in queue:
        directory=out/name;directory.mkdir();candidate=directory/'candidate.csv';candidate.write_bytes(source.read_bytes());assert digest(candidate)==sha
        description='Хайрисламов Тимур | Kaggle '+username+' | target040 '+name+' | sha256 '+sha
        event={'created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'candidate_sha256':sha,'run':str(run),'selected_specification':selection['selected_specification'],'description':description,'uploaded':False,'quota_preflight':limits,'model_sha256':digest(run/'final/model.pkl'),'validation_and_independent_audit_passed':True,'no_automatic_retry':True}
        (directory/'upload-intent.json').write_text(json.dumps(event,indent=2)+'\n')
        try:
            reply=api.competition_submit(str(candidate),description,COMPETITION,quiet=True);ref=int(getattr(reply,'ref',0));assert ref>0
            event.update(uploaded=True,submission_id=ref,status='accepted_pending_score');(directory/'upload-receipt.json').write_text(json.dumps(event,indent=2)+'\n');uploaded.append(event);print(json.dumps({'method':name,'submission_id':ref,'uploaded':True}),flush=True)
        except Exception as error:
            response=getattr(error,'response',None);event.update(status='upload_outcome_requires_reconciliation',exception_type=type(error).__name__,http_status=response.status_code if response is not None else None);(directory/'failure.json').write_text(json.dumps(event,indent=2)+'\n');print(json.dumps({'method':name,'status':event['status'],'http_status':event['http_status']}),flush=True);break
        time.sleep(5)
    time.sleep(10)
    final=[]
    total=limits['numTotal']+len(uploaded)
    for page in range((total+19)//20):
        final.extend(submission_record(x) for x in api.competition_submissions(COMPETITION,page_token=page))
        if page+1<(total+19)//20:time.sleep(3)
    (out/'all-submissions-after.json').write_text(json.dumps(final,indent=2)+'\n')
    for row in final:
        if row['status']=='COMPLETE' and row.get('private_score'):print(json.dumps({'submission_id':row['id'],'private_score':row['private_score'],'description':row['description']}),flush=True)
    best=max((x for x in final if x['status']=='COMPLETE' and x.get('private_score')),key=lambda x:float(x['private_score']))
    result={'finished_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'uploaded_this_resume':len(uploaded),'best_verified':best,'goal_040_reached':float(best['private_score'])>=.4,'observations':final}
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'phase':'complete','uploaded':len(uploaded),'best':best}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--evidence-dir',type=Path,required=True);a=p.parse_args();main(a.root.resolve(),a.evidence_dir.resolve())
