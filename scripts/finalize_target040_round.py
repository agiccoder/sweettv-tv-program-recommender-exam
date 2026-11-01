#!/usr/bin/env python3
"""Wait for an actual round, preserve exact source bytes, independently audit/report."""
import argparse,json,shutil,subprocess,sys,time
from pathlib import Path
from sweettv.data import sha256,dump_json
ROOT=Path(__file__).resolve().parents[1]
def run(a):
    started=time.monotonic()
    while True:
        manifest=json.loads((a.run_dir/'manifest.json').read_text())
        if manifest['status']!='running':break
        if time.monotonic()-started>1200:raise TimeoutError('Bounded wait expired; preserve incomplete run')
        time.sleep(2)
    assert manifest['status']=='complete_local_target040_round' and not manifest['source_changed_during_run']
    for name,value in manifest['source_hashes'].items():
        source=ROOT/name;assert sha256(source)==value,name
        target=a.run_dir/'source-snapshot'/name;assert not target.exists();target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target);assert sha256(target)==value
    checker=a.run_dir/'audit-source.py';assert not checker.exists();shutil.copy2(ROOT/'scripts/audit_target040_round.py',checker)
    dump_json(a.run_dir/'source-snapshot-validation.json',{'passed':True,'source_files':len(manifest['source_hashes']),'audit_source_sha256':sha256(checker),'finalizer_sha256':sha256(__file__)})
    subprocess.run([sys.executable,str(ROOT/'scripts/audit_target040_round.py'),'--run-dir',str(a.run_dir)],check=True)
    subprocess.run([sys.executable,str(ROOT/'scripts/report_target040.py'),'--run-dir',str(a.run_dir),'--output',str(a.report)],check=True)
    print(json.dumps({'phase':'complete_audited_report','run':str(a.run_dir),'report':str(a.report)}),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--report',type=Path,required=True);run(p.parse_args())
