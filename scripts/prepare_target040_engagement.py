"""Read-only, training-cutoff recency features; no model fitting or metrics.

Keep the original interaction seconds/counts unchanged. For each accepted
canonical airing, decay its summed overlap seconds and one watch using its
last clipped positive-overlap endpoint. Aggregate only after the 120s decision.
The graph may use ``seconds`` unchanged; recency affects restart via the
separately named recent_seconds / seconds ratio, not graph construction.
"""
from __future__ import annotations

import argparse
import bisect
import collections
import json
import math
from pathlib import Path
import sqlite3
import time

import pandas as pd

from sweettv.data import channel_segments, dump_json, sha256, timestamp, union_intervals

COLUMNS = ['user_id', 'item_id', 'seconds', 'watches', 'recent_seconds',
           'recent_watches', 'last_event_end', 'capped600', 'capped1800', 'fractional_watches', 'active_days', 'recent_capped600']
DEFAULT_WINDOWS = {
    'development_train': ('2020-03-09T00:00:00', '2020-04-13T00:00:00'),
    'confirmation_train': ('2020-03-09T00:00:00', '2020-05-18T00:00:00'),
    'full': ('2020-03-09T00:00:00', '2020-07-27T00:00:00'),
}
ROOT = Path(__file__).resolve().parents[1]


def _coordinate(value):
    result = timestamp(value) if isinstance(value, str) else float(value)
    if not math.isfinite(result):
        raise ValueError('Window endpoints must be finite')
    return result


def _check_reference(frame, path):
    """Consume the actual original export; demand exact seconds and watches."""
    reference = pd.read_parquet(path)
    keys = ['user_id', 'item_id', 'seconds', 'watches']
    actual = frame[keys].sort_values(keys[:2]).reset_index(drop=True)
    expected = reference[keys].sort_values(keys[:2]).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False, check_exact=True)
    return {'passed': True, 'reference': str(Path(path).resolve()),
            'reference_sha256': sha256(path), 'rows': len(frame),
            'comparison': 'exact keys, seconds and watches; no numerical tolerance'}


def build_engagement_features(db_path, output_dir, source_audit, *, windows=None,
                           half_life_days=28.0, threshold_seconds=120.0,
                           references=None):
    """Export one table per window, opening the existing SQLite DB read-only.

    ``windows`` maps names to (lower-inclusive, upper-exclusive) coordinates.
    ``source_audit`` is the real data-audit mapping (with verified duration
    scales and original-file hashes), not fitted model information.
    ``references`` optionally maps output names to actual original Parquets.
    """
    started = time.perf_counter()
    db_path, output_dir = Path(db_path).resolve(), Path(output_dir).resolve()
    if not db_path.is_file():
        raise FileNotFoundError(db_path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError('Refusing to overwrite an existing recency feature run')
    if not math.isfinite(half_life_days) or half_life_days <= 0:
        raise ValueError('Half-life must be positive and finite')
    if not math.isfinite(threshold_seconds) or threshold_seconds <= 0:
        raise ValueError('Watch threshold must be positive and finite')
    windows = DEFAULT_WINDOWS if windows is None else windows
    bounds = {name: tuple(_coordinate(v) for v in pair) for name, pair in windows.items()}
    if not bounds or any(len(pair) != 2 or pair[0] >= pair[1] for pair in bounds.values()):
        raise ValueError('Require nonempty positive half-open windows')
    scales = {role: float(source_audit['files'][role]['duration_scale_seconds'])
              for role in ['train', 'valid']}
    if any(not math.isfinite(scale) or scale <= 0 for scale in scales.values()):
        raise ValueError('Schedule duration scales must be independently verified positive finite values')
    overall_start = min(pair[0] for pair in bounds.values())
    overall_end = max(pair[1] for pair in bounds.values())
    half_life_seconds = half_life_days * 86400.0
    aggregated = {name: {} for name in bounds}
    stats = {name: collections.Counter() for name in bounds}
    source_paths = [Path(__file__).resolve(), ROOT/'src/sweettv/data.py', ROOT/'src/sweettv/time_features.py']
    source_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in source_paths}
    db_stat = db_path.stat()
    conn = sqlite3.connect(db_path.as_uri() + '?mode=ro', uri=True)
    try:
        conn.execute('PRAGMA query_only=ON')
        channels = [row[0] for row in conn.execute(
            'SELECT DISTINCT channel FROM views WHERE end>? AND start<? AND end>start ORDER BY channel',
            (overall_start, overall_end))]
        for channel in channels:
            airings = []
            for role, scale in scales.items():
                airings.extend((start, start+duration*scale, item) for start,duration,item in conn.execute(
                    'SELECT start,duration,item_id FROM schedule WHERE channel=? AND role=? AND duration>0 ORDER BY start',
                    (channel, role)))
            segments = channel_segments(airings)
            # This immutable search array is allocated ONCE per channel, never
            # per user or per window, unlike the original convenience helper.
            segment_ends = [segment[1] for segment in segments]
            current_user, history = None, []

            def consume(user, intervals):
                if user is None:
                    return
                if not isinstance(user, str):
                    raise ValueError('Raw user IDs must remain strings')
                merged = list(union_intervals(intervals))
                for name, (lower, cutoff) in bounds.items():
                    per_airing = {}
                    for start,end in merged:
                        left_view, right_view = max(start,lower), min(end,cutoff)
                        if right_view <= left_view:
                            continue
                        j = bisect.bisect_right(segment_ends, left_view)
                        while j < len(segments) and segments[j][0] < right_view:
                            s,e,item,airing_start,airing_end,_ = segments[j]
                            left, right = max(left_view,s), min(right_view,e)
                            if item > 0 and right > left:
                                key = (item,airing_start,airing_end)
                                previous = per_airing.get(key)
                                if previous is None:
                                    per_airing[key] = [right-left,right]
                                else:
                                    previous[0] += right-left
                                    previous[1] = max(previous[1],right)
                            j += 1
                    for (item,airing_start,airing_end), (seconds,last_end) in per_airing.items():
                        stats[name]['positive_user_airings'] += 1
                        if seconds < threshold_seconds:
                            stats[name]['below_threshold_user_airings'] += 1
                            continue
                        if last_end > cutoff or last_end <= lower:
                            raise AssertionError('Accepted event endpoint crosses its training cutoff')
                        decay = 2.0 ** (-(cutoff-last_end)/half_life_seconds)
                        cap600,cap1800 = min(seconds/600.,1.),min(seconds/1800.,1.)
                        fraction = min(seconds/(airing_end-airing_start),1.)
                        day = int((last_end-1e-6-lower)//86400.)
                        if day < 0:raise AssertionError('Active-day coordinate precedes training start')
                        mask = 1 << day
                        previous = aggregated[name].get((user,item))
                        if previous is None:
                            aggregated[name][user,item] = [seconds,1,seconds*decay,decay,last_end,cap600,cap1800,fraction,mask,cap600*decay]
                        else:
                            previous[0] += seconds
                            previous[1] += 1
                            previous[2] += seconds*decay
                            previous[3] += decay
                            previous[4] = max(previous[4],last_end)
                            previous[5] += cap600
                            previous[6] += cap1800
                            previous[7] += fraction
                            previous[8] |= mask
                            previous[9] += cap600*decay
                        stats[name]['accepted_user_airings'] += 1
            cursor = conn.execute(
                'SELECT user_id,start,end FROM views WHERE channel=? AND end>? AND start<? AND end>start ORDER BY user_id,start,end',
                (channel,overall_start,overall_end))
            for user,start,end in cursor:
                if user != current_user:
                    consume(current_user,history)
                    current_user, history = user, []
                history.append((start,end))
            consume(current_user,history)
    finally:
        conn.close()
    output_dir.mkdir(parents=True,exist_ok=True)
    audit = {'status':'complete', 'scope':'Training-cutoff engagement features only; no fitting, search or held-out metrics',
             'database':str(db_path), 'sqlite_access':'URI mode=ro plus query_only=ON',
             'database_size_bytes':db_stat.st_size,
             'original_raw_file_hashes':{role:info.get('sha256') for role,info in source_audit['files'].items()},
             'schedule_source_roles':['train','valid'], 'future_schedule_used':False,
             'schedule_duration_scales_seconds':scales, 'half_life_days':half_life_days,
             'threshold_seconds':threshold_seconds,
             'airing_weight':'seconds * 2**(-(window_cutoff - last_clipped_positive_overlap_end)/(half_life_days*86400))',
             'watch_weight':'one identical decay weight per accepted canonical airing',
             'graph_input':'Original unscaled seconds/counts; recent_seconds/seconds is a separately named restart-only scale',
             'last_event_end_definition':'Maximum clipped positive-overlap right endpoint among accepted canonical airings per raw-string user/item',
             'segment_search':'Disjoint canonical channel segments, one cached ends array per channel, bisect per unioned view/window',
             'channels':len(channels), 'source_hashes':source_hashes,'windows':{}}
    references = references or {}
    for name,(lower,cutoff) in bounds.items():
        records = [(user,item,*values[:8],bin(values[8]).count('1'),values[9]) for (user,item),values in sorted(aggregated[name].items())]
        frame = pd.DataFrame.from_records(records,columns=COLUMNS)
        if frame.empty:
            frame = frame.astype({'user_id':'object','item_id':'int64','seconds':'float64','watches':'int64',
                                  'recent_seconds':'float64','recent_watches':'float64','last_event_end':'float64'})
        if not frame.empty:
            if not frame.user_id.map(lambda value:isinstance(value,str)).all():
                raise AssertionError('User ID type drift')
            if (frame.last_event_end > cutoff).any():
                raise AssertionError('Feature event crosses cutoff')
            if (frame.recent_seconds > frame.seconds*(1+1e-14)).any() or (frame.recent_watches > frame.watches*(1+1e-14)).any():
                raise AssertionError('Noncausal or invalid recency weight')
        for signal in ['capped600','capped1800','fractional_watches','active_days','recent_capped600']:
            if not frame.empty and ((frame[signal]<0).any() or (frame[signal]>frame.watches*(1+1e-12)).any()):
                raise AssertionError('Engagement transform exceeds original accepted counts')
        if not frame.empty and ((frame.capped1800>frame.capped600*(1+1e-12)).any() or (frame.recent_capped600>frame.capped600*(1+1e-12)).any()):
            raise AssertionError('Engagement transform order or decay violated')
        parity = _check_reference(frame,references[name]) if name in references else None
        path = output_dir/(name+'.parquet')
        frame.to_parquet(path,index=False)
        audit['windows'][name] = {
            'start_inclusive':lower,'cutoff_exclusive':cutoff,'rows':len(frame),
            'users':int(frame.user_id.nunique()),'items':int(frame.item_id.nunique()),
            'max_event_end':float(frame.last_event_end.max()) if len(frame) else None,
            'min_event_end':float(frame.last_event_end.min()) if len(frame) else None,
            'total_seconds':float(frame.seconds.sum()),'total_watches':int(frame.watches.sum()),
            'total_recent_seconds':float(frame.recent_seconds.sum()),
            'path':str(path), 'sha256':sha256(path), 'airings':dict(stats[name]),
            'exact_original_interaction_parity':parity,
        }
        aggregated[name].clear()
    after_stat = db_path.stat()
    if (after_stat.st_size,after_stat.st_mtime_ns)!=(db_stat.st_size,db_stat.st_mtime_ns):
        raise RuntimeError('Source database changed during read-only feature construction')
    if source_hashes != {str(path.relative_to(ROOT)):sha256(path) for path in source_paths}:
        raise RuntimeError('Feature/source join code changed during construction')
    audit['feature_definition'] = json.loads((ROOT/'evidence/target040-01/round07-engagement-preparation-freeze.json').read_text())['features']
    audit['source_database_size_mtime_unchanged'] = True
    audit['runtime_seconds'] = time.perf_counter()-started
    dump_json(output_dir/'engagement-feature-audit.json',audit)
    return audit



def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'config/target040-round05-preparation.json')
    parser.add_argument('--prepared',type=Path,default=ROOT/'experiments/target040-transfer-data-01')
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    config=json.loads(args.config.read_text())
    windows={b['name']:(b['train_start'],b['train_end']) for b in config['slices']}
    windows['full']=(config['final']['train_start'],config['final']['train_end'])
    references={name:args.prepared/('final' if name=='full' else name)/'train-interactions.parquet' for name in windows}
    base=ROOT/'experiments/competition-20261030-03'
    before=sha256(base/'raw.sqlite')
    result=build_engagement_features(base/'raw.sqlite',args.output_dir,json.loads((base/'audit/data-audit.json').read_text()),windows=windows,references=references)
    assert before==sha256(base/'raw.sqlite')
    dump_json(args.output_dir/'database-hash-verification.json',{'before_and_after_sha256':before,'unchanged':True,'viewing_after_training_cutoffs_used':False,'quality_metrics_computed':False})
    print(json.dumps({'status':result['status'],'runtime_seconds':result['runtime_seconds'],'exact_original_parity_all_windows':all(v['exact_original_interaction_parity']['passed'] for v in result['windows'].values())}))


if __name__=='__main__':
    main()
