#!/usr/bin/env python3
"""Bind a genuine completed best observation to exact server/local CSV bytes."""
import argparse,datetime as dt,hashlib,json
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
def digest(b):return hashlib.sha256(b).hexdigest()
def run(results,out):
    result=read(results);rows=result.get('observations',[])
    best=max((r for r in rows if r['status']=='COMPLETE' and r.get('private_score')),key=lambda r:float(r['private_score']))
    ref=int(best['id']);receipt=None
    for p in (ROOT/'evidence/target040-01').rglob('upload-receipt.json'):
        record=read(p)
        if int(record['submission_id'])==ref:receipt=(p,record);break
    if receipt is None:raise ValueError('No authenticated upload receipt for best ID')
    p,record=receipt;local=p.parent/'candidate.csv';sha=digest(local.read_bytes());assert sha==record['candidate_sha256']
    credentials=read(Path.home()/'.kaggle/kaggle.json')
    response=requests.get('https://www.kaggle.com/api/v1/competitions/submissions/download/'+str(ref),auth=(credentials['username'],credentials['key']),timeout=60)
    response.raise_for_status();assert digest(response.content)==sha
    assert not out.exists();out.mkdir(parents=True);(out/'server-submitted.csv').write_bytes(response.content)
    convenience=ROOT/('submission_target040_best_'+str(best['private_score']).replace('0.','0')+'.csv')
    if convenience.exists():assert convenience.read_bytes()==response.content
    else:convenience.write_bytes(response.content)
    evidence={'observed_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'submission':best,'source_observation':str(results),'source_observation_sha256':digest(results.read_bytes()),'upload_receipt':str(p),'upload_receipt_sha256':digest(p.read_bytes()),'candidate_sha256':sha,'server_download_sha256':digest(response.content),'server_and_local_bytes_identical':True,'model_run':record['run'],'model_sha256':record['model_sha256'],'convenience_csv':str(convenience),'goal_040_reached':float(best['private_score'])>=.4,'evidence_kind':'Authenticated own-account real Kaggle API, not a fabricated webpage or Submissions-page screenshot'}
    (out/'linkage-and-real-score.json').write_text(json.dumps(evidence,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'submission_id':ref,'private_score':best['private_score'],'public_score':best['public_score'],'sha256':sha,'CSV':str(convenience),'server_bytes_match':True}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.results.resolve(),a.output.resolve())
