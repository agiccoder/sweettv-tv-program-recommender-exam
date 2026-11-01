#!/usr/bin/env python3
"""Copy native files byte-for-byte at a stated cutoff, never merge or filter.

Copies stay in private evidence. The two-file course package uses the PDF and
the complete unchanged parent JSONL; child logs remain private and their scope
is disclosed, following PROMPT.txt. No instructor exception is asserted.
No credentials are read, printed, or transformed. Originals remain untouched.
"""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory',default='evidence/native-session-inventory.json')
    p.add_argument('--destination',required=True)
    args=p.parse_args()
    inventory=json.loads(Path(args.inventory).read_text())
    dest=Path(args.destination)
    dest.mkdir(parents=True,exist_ok=False)
    cutoff=dt.datetime.now(dt.timezone.utc).isoformat()
    saved=[]
    for session in inventory['sessions']:
        source=Path(session['path'])
        data=source.read_bytes()
        if not data.endswith(b'\n'):
            raise ValueError('Native file currently has an incomplete final record; retry after it flushes')
        # Validate rather than repairing/reconstructing native content.
        for line in data.splitlines():
            json.loads(line)
        target=dest/source.name
        target.write_bytes(data)
        if target.read_bytes()!=data:
            raise IOError('Native copy differs')
        saved.append({'source':str(source),'copy':str(target),
                      'session_id':session['session_id'],'parent_thread_id':session['parent_thread_id'],
                      'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),
                      'byte_for_byte':True,'unmodified_native_jsonl':True})
    manifest={'cutoff_started_utc':cutoff,'copy_completed_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
              'files':saved,'all_originals_unchanged':True,
              'course_packaging_status':'PDF plus complete unchanged parent native JSONL; separate children retained privately and disclosed, as directed by PROMPT.txt. No special instructor acceptance is asserted.',
              'work_after_cutoff':'Final report rendering/visual inspection and final handoff may occur after this cutoff; later work is not represented by this snapshot.'}
    (dest/'export-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'manifest':str(dest/'export-manifest.json'),
                      'sessions':len(saved),'total_bytes':sum(x['bytes'] for x in saved)},indent=2))


if __name__=='__main__': main()
