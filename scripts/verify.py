#!/usr/bin/env python3
"""Run actual source tests and persist outcomes; no fake competition results."""
import datetime as dt
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from sweettv.cli import environment, source_hashes


class RecordedResult(unittest.TextTestResult):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.outcomes=[]

    def addSuccess(self,test):
        self.outcomes.append({'test':test.id(),'status':'passed'})
        super().addSuccess(test)

    def addFailure(self,test,err):
        self.outcomes.append({'test':test.id(),'status':'failed','details':self._exc_info_to_string(err,test)})
        super().addFailure(test,err)

    def addError(self,test,err):
        self.outcomes.append({'test':test.id(),'status':'error','details':self._exc_info_to_string(err,test)})
        super().addError(test,err)


def main():
    start=time.perf_counter()
    stream=io.StringIO()
    hashes=source_hashes()
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'))
    result=unittest.TextTestRunner(stream=stream,verbosity=2,resultclass=RecordedResult).run(suite)
    log=stream.getvalue()
    evidence=ROOT/'evidence'
    evidence.mkdir(exist_ok=True)
    (evidence/'verification.txt').write_text(log)
    passed=result.wasSuccessful() and hashes==source_hashes()
    record={'scope':'synthetic fixtures and implementation correctness only; no competition performance measured',
            'command':'OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/verify.py',
            'performed_by':'Codex automated test runner, not Timur personal verification',
            'timestamp_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
            'tests_run':result.testsRun,'passed':passed,'failures':len(result.failures),'errors':len(result.errors),
            'skipped':len(result.skipped),'wall_seconds':time.perf_counter()-start,
            'environment':environment(),'source_hashes':hashes,'source_hashes_stable':hashes==source_hashes(),
            'log_sha256':hashlib.sha256(log.encode()).hexdigest(),'tests':result.outcomes}
    (evidence/'verification.json').write_text(json.dumps(record,indent=2)+'\n')
    print(log,end='')
    return 0 if passed else 1


if __name__=='__main__': raise SystemExit(main())
