"""Streaming input audit and channel-specific, half-open interval construction.

Raw inputs are never edited. SQLite keeps the millions of raw rows off the heap.
Schedules are converted into disjoint channel segments: one nonzero active item
labels a segment; multiple distinct labels make it ambiguous and unusable. Zero
IDs are context only, never negative feedback. Overlapping/duplicate views are
unioned within each user/channel before the sorted sweep. A watch threshold is
applied to the union of all fragments of a canonical airing, not each fragment.
"""
from __future__ import annotations

import bisect
import collections
import codecs
import csv
import datetime as dt
import functools
import hashlib
import heapq
import json
import math
import random
import sqlite3
import statistics
import time
import zipfile
from pathlib import Path

import numpy as np

FILES = {
    'views': 'dataset11-30.csv',
    'train': 'export_arh_11-20-final.csv',
    'valid': 'export_arh_21-30-final.csv',
    'future': 'export_arh_31-42-final.csv',
    'template': 'submission.csv',
}
FIELDS = ['channel_title', 'tv_show_title', 'tv_show_category', 'tv_show_genre_1',
          'tv_show_genre_2', 'tv_show_genre_3', 'year_of_production', 'director', 'actors']
EXPECTED_ROWS = {'views': 5821987, 'train': 1074235, 'valid': 1074394, 'future': 1295378,
                 'template': 1260}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def dump_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def discover(raw_dir):
    """Use actual filenames. Fail with inventory rather than inventing aliases."""
    root = Path(raw_dir)
    found = {}
    for role, name in FILES.items():
        paths = sorted(root.rglob(name))
        if len(paths) > 1:
            hashes = {sha256(p) for p in paths}
            if len(hashes) != 1:
                raise ValueError(f'Ambiguous {role} inputs with different hashes: {paths}')
        if paths:
            found[role] = paths[0]
    if len(found) != len(FILES):
        missing = [name for role, name in FILES.items() if role not in found]
        raise FileNotFoundError('Missing competition inputs: ' + ', '.join(missing) +
                                f'. Place the authorized Kaggle download under {root}.')
    return found


def extract_archive(archive, destination):
    """Extract an explicitly supplied ZIP; no traversal, symlinks or overwrite."""
    dest = Path(destination).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            target = (dest / info.filename).resolve()
            if dest != target and dest not in target.parents:
                raise ValueError('Unsafe archive path')
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('Archive symlink is not permitted')
            if target.exists() and not info.is_dir():
                raise FileExistsError(f'Refusing overwrite of {target}')
        z.extractall(dest)
    return {'archive_sha256': sha256(archive), 'destination': str(dest)}


def csv_format(path):
    with open(path, 'rb') as f:
        block = f.read(131072)
    encoding = 'utf-8-sig'
    try:
        text = codecs.getincrementaldecoder(encoding)().decode(block, final=False)
    except UnicodeDecodeError as e:
        # A sample may end in the middle of a valid UTF-8 multibyte character.
        try:
            text = block[:-4].decode(encoding)
        except UnicodeDecodeError:
            raise ValueError(f'{path}: encoding is not verified UTF-8; inspect explicitly') from e
    dialect = csv.Sniffer().sniff(text, delimiters=',;\t|')
    return encoding, dialect


def rows(path):
    encoding, dialect = csv_format(path)
    with open(path, encoding=encoding, newline='') as f:
        reader = csv.DictReader(f, dialect=dialect)
        for i, row in enumerate(reader, 2):
            if None in row or any(v is None for v in row.values()):
                raise ValueError(f'{path}:{i}: ragged CSV record')
            yield row


@functools.lru_cache(maxsize=300000)
def timestamp(value):
    s = value.strip()
    if not s:
        raise ValueError('Missing timestamp')
    try:
        number = float(s)
    except ValueError:
        t = parse_datetime(s)
        # Naive timestamps share a floating wall-clock coordinate, not a claim
        # that the source is UTC. Mixed naive/aware sources are rejected below.
        t = t.replace(tzinfo=dt.timezone.utc) if t.tzinfo is None else t
        return t.timestamp()
    if not math.isfinite(number) or abs(number) < 1e8:
        raise ValueError(f'Unrecognized timestamp representation: {s}')
    if abs(number) > 1e14:
        return number / 1e6
    if abs(number) > 1e11:
        return number / 1000
    return number


def time_kind(value):
    try:
        float(value)
        return 'numeric_epoch'
    except ValueError:
        t = parse_datetime(value)
        return 'naive_wall_clock' if t.tzinfo is None else 'explicit_offset'


@functools.lru_cache(maxsize=300000)
def parse_datetime(value):
    """Two formats observed in the real archive; never infer month/day order."""
    try:
        return dt.datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError:
        return dt.datetime.strptime(value, '%d.%m.%Y %H:%M:%S')


def integer(value):
    s = value.strip()
    if not s or not s.lstrip('-').isdigit():
        raise ValueError(f'Expected exact integer identifier, got {value!r}')
    return int(s)


def quantiles(values):
    return {str(q): float(np.quantile(values, q)) for q in [0, .1, .25, .5, .75, .9, .99, 1]} if values else {}


class Reservoir:
    def __init__(self, seed=20261030, size=100000):
        self.values, self.n, self.size, self.rng = [], 0, size, random.Random(seed)

    def add(self, x):
        self.n += 1
        if len(self.values) < self.size:
            self.values.append(x)
        else:
            j = self.rng.randrange(self.n)
            if j < self.size:
                self.values[j] = x


def infer_duration_unit(conn, role):
    """Compare raw duration with next distinct start on SAME channel, not views.

    A winner must explain at least 25% of sampled endpoints within two seconds
    and beat the runner up by >=0.15. Otherwise inspection/explicit override is
    mandatory. Gaps and overlaps remain separately audited after conversion.
    """
    sample = conn.execute('''WITH s AS (
        SELECT channel,start,MAX(duration) AS d FROM schedule WHERE role=?
        GROUP BY channel,start), g AS (
        SELECT d,LEAD(start) OVER(PARTITION BY channel ORDER BY start)-start AS gap FROM s)
        SELECT d,gap FROM g WHERE d>0 AND gap>0 LIMIT 100000''', (role,)).fetchall()
    scores = {}
    for name, scale in [('seconds', 1), ('minutes', 60), ('milliseconds', .001)]:
        errors = [abs(d * scale - gap) for d, gap in sample]
        scores[name] = {'scale_seconds': scale, 'sample_size': len(sample),
                        'endpoint_match_fraction': sum(e <= 2 for e in errors) / max(1, len(errors)),
                        'median_absolute_endpoint_error_seconds': statistics.median(errors) if errors else None}
    ordered = sorted(scores, key=lambda k: -scores[k]['endpoint_match_fraction'])
    best, second = [scores[k]['endpoint_match_fraction'] for k in ordered[:2]]
    if best < .25 or best - second < .15:
        return None, scores
    return scores[ordered[0]]['scale_seconds'], scores


def import_data(raw_dir, db_path, evidence_dir, duration_units=None):
    start_clock = time.perf_counter()
    files = discover(raw_dir)  # Do not create a database on missing raw inputs.
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        raise FileExistsError(f'{db_path} already exists; use a new run directory')
    conn = sqlite3.connect(db_path)
    conn.executescript('''PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;
        PRAGMA temp_store=FILE;
        CREATE TABLE views(user_id TEXT,channel INTEGER,start REAL,end REAL,duration REAL);
        CREATE TABLE schedule(role TEXT,channel INTEGER,start REAL,duration REAL,item_id INTEGER,metadata TEXT);
    ''')
    audit = {'status': 'input_audit', 'files': {}, 'seed': 20261030,
             'time_zone_note': 'Naive timestamps retain source wall-clock order; no source timezone inferred.'}
    kinds = set()
    for role, path in files.items():
        encoding, dialect = csv_format(path)
        stats = {'path': str(path.resolve()), 'sha256': sha256(path), 'bytes': path.stat().st_size,
                 'encoding': encoding, 'delimiter': dialect.delimiter, 'quotechar': dialect.quotechar,
                 'rows': 0, 'missing': collections.Counter(), 'time_kinds': collections.Counter()}
        if role == 'template':
            template = list(rows(path))
            stats['rows'] = len(template)
            stats['columns'] = list(template[0]) if template else []
            stats['delta_from_instructor_rows'] = len(template) - EXPECTED_ROWS[role]
            audit['files'][role] = stats
            continue
        sample, pending = Reservoir(), []
        invalid, mismatch, labelled = 0, 0, 0
        minimum, maximum = float('inf'), float('-inf')
        for row in rows(path):
            if not stats['rows']:
                stats['columns'] = list(row)
                stats['first_record'] = row
            stats['rows'] += 1
            for key, value in row.items():
                if value.strip() == '':
                    stats['missing'][key] += 1
            kind = time_kind(row['start_time'])
            stats['time_kinds'][kind] += 1
            kinds.add(kind)
            s = timestamp(row['start_time'])
            minimum, maximum = min(minimum, s), max(maximum, s)
            if role == 'views':
                end_kind = time_kind(row['stop_time'])
                kinds.add(end_kind)
                e, d = timestamp(row['stop_time']), float(row['duraton'])
                if not math.isfinite(d):
                    raise ValueError('View duraton must be finite')
                invalid += int(e <= s or d < 0)
                mismatch += int(abs(e - s - d) > 1)
                sample.add(e - s)
                pending.append((row['user_id'], integer(row['vsetv_id']), s, e, d))
            else:
                item, d = integer(row['tv_show_id']), float(row['duration'])
                if item < 0 or not math.isfinite(d):
                    raise ValueError('Invalid item/duration')
                labelled += item != 0
                invalid += d <= 0
                sample.add(d)
                metadata = {k: row.get(k, '') for k in FIELDS}
                pending.append((role, integer(row['channel_id']), s, d, item,
                                json.dumps(metadata, ensure_ascii=False)))
            if len(pending) >= 20000:
                conn.executemany('INSERT INTO views VALUES(?,?,?,?,?)' if role == 'views' else
                                 'INSERT INTO schedule VALUES(?,?,?,?,?,?)', pending)
                conn.commit()
                pending = []
        if pending:
            conn.executemany('INSERT INTO views VALUES(?,?,?,?,?)' if role == 'views' else
                             'INSERT INTO schedule VALUES(?,?,?,?,?,?)', pending)
        conn.commit()
        stats.update({'invalid_nonpositive_intervals': invalid,
                      'start_min': minimum, 'start_max': maximum,
                      'sampled_duration_quantiles': quantiles(sample.values),
                      'duration_sample_size': len(sample.values),
                      'delta_from_instructor_rows': stats['rows'] - EXPECTED_ROWS[role]})
        if role == 'views':
            stats['duraton_vs_endpoint_mismatches_gt_1s'] = mismatch
        else:
            stats['labelled_rows'] = labelled
            stats['unlabelled_rows'] = stats['rows'] - labelled
            stats['labelled_fraction'] = labelled / max(1, stats['rows'])
        audit['files'][role] = stats
    if 'naive_wall_clock' in kinds and kinds != {'naive_wall_clock'}:
        raise ValueError(f'Mixed naive/absolute times require explicit alignment: {kinds}')
    audit['time_kinds'] = sorted(kinds)
    conn.executescript('CREATE INDEX schedule_channel ON schedule(role,channel,start); '
                       'CREATE INDEX view_channel ON views(channel,user_id,start,end);')
    for role in ['train', 'valid', 'future']:
        scale, scores = infer_duration_unit(conn, role)
        override = (duration_units or {}).get(role)
        if override is not None:
            scale = float(override)
        audit['files'][role]['duration_unit_test'] = scores
        audit['files'][role]['duration_scale_seconds'] = scale
        audit['files'][role]['duration_override'] = override
        if scale is None:
            dump_json(Path(evidence_dir) / 'data-audit.json', audit)
            raise ValueError(f'Ambiguous {role} duration unit. Inspect data-audit.json and set explicit units.')
        audit['files'][role]['distinct_programmes_including_zero'] = conn.execute(
            'SELECT count(DISTINCT item_id) FROM schedule WHERE role=?', (role,)).fetchone()[0]
        audit['files'][role]['channels'] = conn.execute(
            'SELECT count(DISTINCT channel) FROM schedule WHERE role=?', (role,)).fetchone()[0]
        audit['files'][role]['duplicate_airing_rows'] = conn.execute('''SELECT SUM(n-1) FROM
            (SELECT COUNT(*) n FROM schedule WHERE role=? GROUP BY channel,start,duration,item_id)''',
            (role,)).fetchone()[0] or 0
    audit['view_distinct_users'] = conn.execute('SELECT count(DISTINCT user_id) FROM views').fetchone()[0]
    audit['view_duplicate_rows'] = conn.execute('''SELECT SUM(n-1) FROM
        (SELECT COUNT(*) n FROM views GROUP BY user_id,channel,start,end)''').fetchone()[0] or 0
    audit['runtime_seconds'] = time.perf_counter() - start_clock
    audit['status'] = 'complete'
    dump_json(Path(evidence_dir) / 'data-audit.json', audit)
    conn.close()
    return audit


def union_intervals(intervals):
    """Union half-open intervals, also merging adjacent fragments."""
    current = None
    for s, e in sorted(intervals):
        if e <= s:
            continue
        if current is not None and s <= current[1]:
            current = (current[0], max(current[1], e))
        else:
            if current is not None:
                yield current
            current = (s, e)
    if current is not None:
        yield current


def channel_segments(airings):
    """Return disjoint segments (start,end,item,airing_start,airing_end,reason).

    Same-item duplicate/overlapping airings are unioned first. A same-item repeat
    after a gap remains a distinct airing. Exactly adjoining rows remain separate
    airings (their shared boundary never creates a positive overlap).
    """
    by_item = collections.defaultdict(list)
    for s, e, item in airings:
        if e > s:
            by_item[item].append((s, e))
    canonical = []
    for item, intervals in sorted(by_item.items()):
        merged = None
        for s, e in sorted(set(intervals)):
            if merged and s < merged[1]:
                merged = (merged[0], max(e, merged[1]))
            else:
                if merged:
                    canonical.append((*merged, item))
                merged = (s, e)
        if merged:
            canonical.append((*merged, item))
    events = collections.defaultdict(list)
    for key, (s, e, item) in enumerate(canonical):
        events[s].append((1, key))
        events[e].append((-1, key))
    active, previous, result = set(), None, []
    for t in sorted(events):
        if previous is not None and t > previous and active:
            labels = {canonical[k][2] for k in active if canonical[k][2] != 0}
            if len(labels) == 1:
                item = next(iter(labels))
                key = next(k for k in active if canonical[k][2] == item)
                s, e, _ = canonical[key]
                record = (previous, t, item, s, e, 'labelled')
            else:
                reason = 'ambiguous_labels' if len(labels) > 1 else 'unlabelled'
                record = (previous, t, 0, previous, t, reason)
            # Coalesce harmless event splits in one airing for faster joining.
            if result and result[-1][1] == previous and result[-1][2:] == record[2:]:
                result[-1] = (result[-1][0], t, *record[2:])
            else:
                result.append(record)
        for direction, key in events[t]:
            if direction == -1:
                active.remove(key)
        for direction, key in events[t]:
            if direction == 1:
                active.add(key)
        previous = t
    return result


def overlap_segments(views, segments):
    """Sorted interval join, O(V log S + actual overlapping pairs)."""
    ends = [s[1] for s in segments]
    for start, end in union_intervals(views):
        j = bisect.bisect_right(ends, start)
        while j < len(segments) and segments[j][0] < end:
            seg = segments[j]
            left, right = max(start, seg[0]), min(end, seg[1])
            if right > left:
                yield (start, end, seg, left, right)
            j += 1


def build_interactions(db_path, audit, output_dir, threshold_seconds=120, calendar_boundaries=None):
    """Freeze protocol from observed window starts, clip crossing intervals.

    The 120-second absolute threshold is a predeclared engineering choice,
    assessed descriptively against observed distributions, never tuned by AP.
    No normalized watch percentage is assumed. No future views are read.
    """
    clock = time.perf_counter()
    conn = sqlite3.connect(db_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    observed_starts = {r: audit['files'][r]['start_min'] for r in ['train', 'valid', 'future']}
    starts = ({r: timestamp(calendar_boundaries[r]) for r in ['train','valid','future']}
              if calendar_boundaries else observed_starts)
    if not starts['train'] < starts['valid'] < starts['future']:
        raise ValueError(f'Observed schedules do not support ordered windows: {starts}')
    full_start, cutoff, future_start = [starts[r] for r in ['train', 'valid', 'future']]
    # Minima must separate nominal files; overlapping early records need a
    # manually documented calendar boundary rather than silently guessing.
    if not calendar_boundaries and (audit['files']['train']['start_max'] >= cutoff or audit['files']['valid']['start_max'] >= future_start):
        raise ValueError('Schedule files overlap in start dates. Set and document an observed calendar protocol.')
    future_end = timestamp(calendar_boundaries['end']) if calendar_boundaries and calendar_boundaries.get('end') else None
    candidates, metadata = {}, {}
    for role in ['train', 'valid', 'future']:
        lo = starts[role]
        hi = {'train':cutoff, 'valid':future_start, 'future':future_end}.get(role)
        source_roles = ['future'] if role=='future' else ['train','valid']
        pieces=[]
        for source_role in source_roles:
            scale=audit['files'][source_role]['duration_scale_seconds']
            query='SELECT item_id,metadata,start,channel FROM schedule WHERE role=? AND item_id>0 AND duration>0 AND start+duration*?>?'
            params=[source_role,scale,lo]
            if hi is not None:
                query+=' AND start<?'
                params.append(hi)
            pieces.extend(conn.execute(query,params))
        pieces.sort(key=lambda r:(r[2],r[3],r[0],r[1]))
        candidates[role]=sorted({r[0] for r in pieces})
        if len(candidates[role]) < 5:
            raise ValueError('Fewer than five labelled candidates')
        # Earliest nonempty field by (time,channel,item) is deterministic. No
        # future metadata leaks into either training or validation profiles.
        meta = {}
        for item, raw, _, _ in pieces:
            record = json.loads(raw)
            prior = meta.setdefault(item, {key: '' for key in FIELDS})
            for key, value in record.items():
                if not prior.get(key) and value:
                    prior[key] = value
        metadata[role] = meta
    channels = [r[0] for r in conn.execute('SELECT DISTINCT channel FROM views ORDER BY channel')]
    windows = ['train', 'valid', 'full']
    totals = {w: collections.Counter() for w in windows}
    users = {w: collections.defaultdict(set) for w in windows}
    aggregated = {w: collections.defaultdict(lambda: [0., 0]) for w in windows}
    reservoir = {w: Reservoir() for w in windows}
    examples = []
    all_users = set()
    for channel in channels:
        airings = []
        for role in ['train', 'valid']:
            scale = audit['files'][role]['duration_scale_seconds']
            for s, d, item in conn.execute('SELECT start,duration,item_id FROM schedule WHERE channel=? AND role=? AND duration>0 ORDER BY start', (channel, role)):
                airings.append((s, s + d * scale, item))
        segments = channel_segments(airings)
        cursor = conn.execute('SELECT user_id,start,end FROM views WHERE channel=? AND end>? AND start<? AND end>start ORDER BY user_id,start,end',
                              (channel, full_start, future_start))
        current, history = None, []

        def consume(user, history):
            if user is None:
                return
            all_users.add(user)
            merged = list(union_intervals(history))
            for window, lo, hi in [('train', full_start, cutoff), ('valid', cutoff, future_start),
                                   ('full', full_start, future_start)]:
                clipped = [(max(s, lo), min(e, hi)) for s, e in merged if s < hi and e > lo]
                if not clipped:
                    continue
                users[window]['any_view'].add(user)
                totals[window]['view_seconds_after_union'] += sum(e-s for s, e in clipped)
                totals[window]['merged_view_intervals'] += len(clipped)
                totals[window]['raw_view_intervals'] += sum(s < hi and e > lo for s,e in history)
                per_airing = collections.defaultdict(float)
                matched = 0
                for vs, ve, seg, left, right in overlap_segments(clipped, segments):
                    s, e, item, astart, aend, reason = seg
                    seconds = right-left
                    matched += seconds
                    totals[window][reason+'_seconds'] += seconds
                    totals[window][reason+'_fragments'] += 1
                    users[window][reason].add(user)
                    if item:
                        per_airing[(item, astart, aend)] += seconds
                    if len(examples) < 120 and (item or len(examples) < 30):
                        examples.append({'window': window, 'user_id': user, 'channel_id': channel,
                                         'view_start': vs, 'view_end': ve, 'airing_start': astart,
                                         'airing_end': aend, 'overlap_seconds': seconds,
                                         'item_id': item, 'reason': reason,
                                         'airing_key': [user, channel, item, astart, aend]})
                totals[window]['outside_schedule_seconds'] += sum(e-s for s,e in clipped)-matched
                for (item, astart, aend), seconds in per_airing.items():
                    reservoir[window].add(seconds)
                    totals[window]['positive_user_airings_before_threshold'] += 1
                    if seconds >= threshold_seconds:
                        aggregated[window][user,item][0] += seconds
                        aggregated[window][user,item][1] += 1
                        users[window]['usable'].add(user)
                    else:
                        totals[window]['below_threshold_seconds'] += seconds
                        totals[window]['below_threshold_user_airings'] += 1
                for example in examples:
                    if example['window'] == window and example['user_id'] == user and example['channel_id'] == channel:
                        key = tuple(example['airing_key'][2:])
                        total = per_airing.get(key,0.)
                        example['aggregated_airing_overlap_seconds'] = total
                        example['watched'] = bool(example['item_id'] and total >= threshold_seconds)

        for user, s, e in cursor:
            if user != current:
                consume(current, history)
                current, history = user, []
            history.append((s,e))
        consume(current, history)
    result = {}
    for window in windows:
        records = [{'user_id': u, 'item_id': i, 'seconds': seconds, 'watches': watches}
                   for (u,i),(seconds,watches) in sorted(aggregated[window].items())]
        import pandas as pd
        pd.DataFrame(records, columns=['user_id','item_id','seconds','watches']).to_parquet(output_dir / f'{window}-interactions.parquet', index=False)
        stats = dict(totals[window])
        stats['users'] = {k: len(v) for k,v in users[window].items()}
        stats['users_without_labelled_time'] = len(users[window]['any_view'] - users[window]['labelled'])
        stats['users_without_usable_interactions'] = len(users[window]['any_view'] - users[window]['usable'])
        stats['usable_user_item_pairs'] = len(records)
        stats['overlap_duration_quantiles_reservoir'] = quantiles(reservoir[window].values)
        result[window] = stats
    for example in examples:
        example.pop('airing_key', None)
    protocol = {'boundaries': starts, 'observed_file_minima':observed_starts,
                'explicit_calendar_boundaries':calendar_boundaries,
                'calendar_rationale':'Explicit week-start midnights after inspecting schedule spillovers; clip rows and features to actual windows.' if calendar_boundaries else 'Nonoverlapping observed file minima.',
                'future_end':future_end,
                'timestamp_display': {k: dt.datetime.fromtimestamp(v,dt.timezone.utc).isoformat() for k,v in starts.items()},
                'display_note': 'UTC display of stored coordinate; naive input values have unspecified source timezone.',
                'windows_half_open': {'train': [full_start,cutoff], 'valid': [cutoff,future_start]},
                'threshold_seconds': threshold_seconds, 'threshold_selected_by_validation': False,
                'repeat_policy': 'allow', 'future_views_used': False,
                'boundary_policy': 'clip views/overlap to each half-open window; threshold fragments separately per window',
                'overlap_policy': 'union views by user/channel; union same-item overlapping airings; discard seconds with conflicting nonzero labels',
                'unlabelled_policy': 'discard as preferences, retain audit context only',
                'runtime_seconds': time.perf_counter()-clock}
    dump_json(output_dir/'protocol.json', protocol)
    dump_json(output_dir/'join-audit.json', result)
    dump_json(output_dir/'join-examples.json', examples)
    dump_json(output_dir/'candidates.json', candidates)
    dump_json(output_dir/'metadata.json', metadata)
    dump_json(output_dir/'observed-users.json', sorted(all_users))
    conn.close()
    return protocol, result
