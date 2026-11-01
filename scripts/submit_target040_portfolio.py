#!/usr/bin/env python3
"""Submit an audited predeclared portfolio; three bounded upload workers."""
import argparse,concurrent.futures,datetime as dt,json,subprocess,sys,time
from pathlib import Path
from target040_kaggle import digest,read,submission_record,validate_run,COMPETITION
ROOT=Path(__file__).resolve().parents[1]

def main(root,destination):
    if destination.exists():raise FileExistsError(destination)
    audit=read(root/'portfolio-independent-audit.json');assert audit['passed']
    destination.mkdir(parents=True)
    known={}
    for p in (ROOT/'evidence/target040-01').rglob('upload-receipt.json'):
        x=read(p);known[x['candidate_sha256']]=x.get('submission_id')
    history=ROOT/'evidence/target040-01/historical-submissions/linkage-register.json'
    for x in read(history):known[x['downloaded_sha256']]=x['submission_id']
    queue=[];skipped=[]
    for item in audit['candidates']:
        path=root/'candidates'/item['method'];source,sha,_=validate_run(path)
        if sha in known:skipped.append({'method':item['method'],'sha256':sha,'already_submitted_or_queued':known[sha]});continue
        known[sha]='queued';queue.append((item['method'],path,sha))
    (destination/'frozen-upload-queue.json').write_text(json.dumps({'created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'queue':[{'method':n,'path':str(p),'sha256':h} for n,p,h in queue],'skipped_identical':skipped,'workers':3,'script_sha256':digest(__file__),'portfolio_config_sha256':digest(root/'frozen-config.json')},indent=2)+'\n')
    def submit(task):
        name,path,sha=task;out=destination/name
        with (destination/(name+'-console.txt')).open('x') as stream:
            result=subprocess.run([sys.executable,str(ROOT/'scripts/target040_kaggle.py'),'--mode','submit','--run-dir',str(path),'--evidence-dir',str(out)],stdout=stream,stderr=subprocess.STDOUT,cwd=ROOT)
        receipt=read(out/'upload-receipt.json') if (out/'upload-receipt.json').exists() else {}
        observation=read(out/'real-submission-observation.json') if (out/'real-submission-observation.json').exists() else {}
        return {'method':name,'sha256':sha,'exit_code':result.returncode,'submission_id':receipt.get('submission_id'),'observation':observation}
    records=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures=[executor.submit(submit,task) for task in queue]
        for future in concurrent.futures.as_completed(futures):
            row=future.result();records.append(row);print(json.dumps(row),flush=True)
    from kaggle import api
    for attempt in range(6):
        observed={int(x.ref):submission_record(x) for x in api.competition_submissions(COMPETITION,page_size=200)}
        pending=[]
        for row in records:
            if row['submission_id'] in observed:row['observation']=observed[row['submission_id']]
            if row['submission_id'] and row['observation'].get('status')!='COMPLETE':pending.append(row['submission_id'])
        p=destination/('poll-%02d.json'%attempt);p.write_text(json.dumps({'observed_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'records':records,'pending':pending},indent=2)+'\n')
        if not pending:break
        time.sleep(5)
    scored=[r for r in records if r['observation'].get('status')=='COMPLETE' and r['observation'].get('private_score')]
    best=max(scored,key=lambda x:float(x['observation']['private_score'])) if scored else None
    final={'finished_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'records':records,'skipped_identical':skipped,'best_verified':best,'goal_private_040_reached':bool(best and float(best['observation']['private_score'])>=.4),'queue_frozen_before_batch_scores':True}
    (destination/'batch-results.json').write_text(json.dumps(final,indent=2)+'\n');print(json.dumps({'phase':'batch_complete','uploaded':len(scored),'best':best}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--evidence-dir',type=Path,required=True);a=p.parse_args();main(a.root.resolve(),a.evidence_dir.resolve())
