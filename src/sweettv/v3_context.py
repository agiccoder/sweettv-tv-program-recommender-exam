"""Cutoff-only raw channel/time profiles and schedule availability, without item fabrication."""
from __future__ import annotations

from collections import defaultdict
import bisect
import json
from pathlib import Path
import sqlite3
import time

import numpy as np
from scipy import sparse

from .data import channel_segments, dump_json, sha256, timestamp, union_intervals


def calendar_parts(start, end, hour_width=4):
    """Split a naive wall-clock coordinate at calendar-aligned hour buckets.

    Day zero (1970-01-01) is Thursday. Arithmetic never converts source timezone.
    """
    width = hour_width * 3600
    if hour_width <= 0 or 24 % hour_width:
        raise ValueError("Hour width must divide 24")
    current = float(start)
    while current < end:
        right = min(float(end), (int(current // width) + 1) * width)
        day = int(current // 86400)
        weekday = (day + 3) % 7
        hour_bin = int((current - day * 86400) // width)
        yield weekday, hour_bin, right - current
        current = right


def interval_profile(intervals, lower, cutoff, hour_width=4):
    """Union one user/channel's input before clipping and calendar splitting."""
    counts = defaultdict(float)
    for start, end in union_intervals(intervals):
        left, right = max(start, lower), min(end, cutoff)
        if right > left:
            for weekday, hour, seconds in calendar_parts(left, right, hour_width):
                counts[weekday, hour] += seconds
    return dict(counts)


def availability_matrix(airings_by_channel, candidates, channels, lower, upper, hour_width=4):
    """Unique non-conflicting schedule seconds, normalized to a duration mean.

    No metadata, viewer labels or future vocabulary is read. Channel columns are
    supplied from the *training* profile. Unknown channels get zero affinity;
    their broadcast duration remains in the item's normalization denominator.
    """
    items = sorted(set(int(item) for item in candidates))
    if any(item <= 0 for item in items):
        raise ValueError("Availability candidates must be positive programme IDs")
    item_index = {item: i for i, item in enumerate(items)}
    channel_index = {int(channel): i for i, channel in enumerate(channels)}
    per_day = 24 // hour_width
    quantities, totals = defaultdict(float), np.zeros(len(items), dtype=np.float64)
    audit = defaultdict(float)
    for channel, airings in sorted(airings_by_channel.items()):
        clipped = [(max(s, lower), min(e, upper), int(item)) for s, e, item in airings
                   if s < upper and e > lower and e > s]
        for start, end, item, _, _, reason in channel_segments(clipped):
            audit[reason + "_seconds"] += end - start
            index = item_index.get(item)
            if index is None or item <= 0:
                continue
            totals[index] += end - start
            if channel not in channel_index:
                audit["positive_seconds_on_unknown_training_channels"] += end - start
                continue
            offset = channel_index[channel] * 7 * per_day
            for weekday, hour, seconds in calendar_parts(start, end, hour_width):
                quantities[index, offset + weekday * per_day + hour] += seconds
    keys = sorted(quantities)
    matrix = sparse.csr_matrix(([quantities[key] / totals[key[0]] for key in keys],
                                ([key[0] for key in keys], [key[1] for key in keys])),
                               shape=(len(items), len(channels) * 7 * per_day), dtype=np.float64)
    audit.update({"candidate_count": len(items), "candidates_with_positive_airtime": int(np.count_nonzero(totals)),
                  "candidates_with_known_channel_features": int(np.count_nonzero(matrix.getnnz(axis=1))),
                  "unique_labelled_airtime_seconds": float(totals.sum()), "matrix_nonzeros": matrix.nnz})
    return matrix, items, totals, dict(audit)


class ContextState:
    def __init__(self, counts, users, channels, lower, cutoff, smoothing_seconds=21600., hour_width=4):
        self.counts = counts.tocsr().astype(np.float64)
        self.users, self.channels = list(users), list(channels)
        self.user_index = {user: i for i, user in enumerate(users)}
        self.lower, self.cutoff = float(lower), float(cutoff)
        self.smoothing_seconds, self.hour_width = float(smoothing_seconds), int(hour_width)
        self.totals = np.asarray(self.counts.sum(axis=1)).ravel()
        self.global_profile = np.asarray(self.counts.sum(axis=0)).ravel()
        total = self.global_profile.sum()
        if total:
            self.global_profile /= total
        if self.counts.shape != (len(self.users), len(self.channels) * 7 * (24 // hour_width)):
            raise ValueError("Context dimensions differ from raw users/training channels")
        if any(not isinstance(user, str) for user in self.users):
            raise ValueError("Raw user IDs must remain strings")
        if np.any(~np.isfinite(self.counts.data)) or np.any(self.counts.data < 0):
            raise ValueError("Context counts must be finite nonnegative seconds")

    def scores(self, user, exposure):
        index = self.user_index.get(user)
        if index is None or self.totals[index] <= 0:
            return None  # Genuine no-history fallback, never an invented global preference.
        own = (exposure @ self.counts.getrow(index).T).toarray().ravel()
        global_score = np.asarray(exposure @ self.global_profile).ravel()
        return (own + self.smoothing_seconds * global_score) / (self.totals[index] + self.smoothing_seconds)

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        sparse.save_npz(directory / "counts.npz", self.counts)
        dump_json(directory / "profile.json", {"users": self.users, "channels": self.channels,
                  "lower": self.lower, "cutoff": self.cutoff, "smoothing_seconds": self.smoothing_seconds,
                  "hour_width": self.hour_width, "raw_total_seconds": float(self.totals.sum())})

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        meta = json.loads((directory / "profile.json").read_text())
        return cls(sparse.load_npz(directory / "counts.npz"), meta["users"], meta["channels"],
                   meta["lower"], meta["cutoff"], meta["smoothing_seconds"], meta["hour_width"])


def collect_contexts(database, output, windows, config, source_audit):
    """One channel-sorted read-only pass for all fixed training cutoffs."""
    started = time.perf_counter()
    database, output = Path(database).resolve(), Path(output)
    if output.exists():
        raise FileExistsError(output)
    bounds = {name: (timestamp(lo), timestamp(hi)) for name, (lo, hi) in windows.items()}
    counts = {name: defaultdict(float) for name in windows}
    users, channel_sets = {name: set() for name in windows}, {name: set() for name in windows}
    full_name = max(bounds, key=lambda name: bounds[name][1])
    audit, before = defaultdict(float), database.stat()
    connection = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)
    connection.execute("PRAGMA query_only=ON")
    lower, upper = min(pair[0] for pair in bounds.values()), max(pair[1] for pair in bounds.values())
    channels = [row[0] for row in connection.execute("SELECT DISTINCT channel FROM views WHERE start<? AND end>? AND end>start ORDER BY channel", (upper, lower))]
    try:
        for channel in channels:
            airings = []
            for role in ["train", "valid"]:
                scale = source_audit["files"][role]["duration_scale_seconds"]
                airings.extend((s, s + d * scale, item) for s, d, item in connection.execute(
                    "SELECT start,duration,item_id FROM schedule WHERE channel=? AND role=? AND duration>0 ORDER BY start", (channel, role)))
            segments = channel_segments(airings)
            ends = [segment[1] for segment in segments]

            def consume(user, intervals):
                if user is None:
                    return
                if not isinstance(user, str):
                    raise ValueError("Raw user ID lost its string type")
                merged = list(union_intervals(intervals))
                for name, (lo, cutoff) in bounds.items():
                    profile = interval_profile(merged, lo, cutoff, config["hour_bin_width"])
                    if profile:
                        users[name].add(user)
                        channel_sets[name].add(channel)
                        for (weekday, hour), seconds in profile.items():
                            counts[name][user, channel, weekday, hour] += seconds
                    if name == full_name:
                        for s, e in merged:
                            left, right = max(s, lo), min(e, cutoff)
                            if right <= left:
                                continue
                            audit["view_seconds_after_union"] += right - left
                            matched, j = 0., bisect.bisect_right(ends, left)
                            while j < len(segments) and segments[j][0] < right:
                                segment = segments[j]
                                seconds = min(right, segment[1]) - max(left, segment[0])
                                if seconds > 0:
                                    audit[segment[-1] + "_seconds"] += seconds
                                    matched += seconds
                                j += 1
                            audit["outside_schedule_seconds"] += right - left - matched
            current, history = None, []
            for user, s, e in connection.execute("SELECT user_id,start,end FROM views WHERE channel=? AND start<? AND end>? AND end>start ORDER BY user_id,start,end", (channel, upper, lower)):
                if user != current:
                    consume(current, history)
                    current, history = user, []
                history.append((s, e))
            consume(current, history)
    finally:
        connection.close()
    output.mkdir(parents=True)
    result = {"status": "complete", "future_views_used": False, "sqlite": "mode=ro/query_only=ON",
              "full_raw_time_audit": dict(audit), "windows": {}}
    for name, (lo, cutoff) in bounds.items():
        ordered_users, ordered_channels = sorted(users[name]), sorted(channel_sets[name])
        ui, ci = {u: i for i, u in enumerate(ordered_users)}, {c: i for i, c in enumerate(ordered_channels)}
        keys = sorted(counts[name])
        per_day = 24 // config["hour_bin_width"]
        matrix = sparse.csr_matrix(([counts[name][key] for key in keys],
                    ([ui[key[0]] for key in keys], [(ci[key[1]] * 7 + key[2]) * per_day + key[3] for key in keys])),
                    shape=(len(ordered_users), len(ordered_channels) * 7 * per_day))
        state = ContextState(matrix, ordered_users, ordered_channels, lo, cutoff,
                             config["smoothing_seconds"], config["hour_bin_width"])
        state.save(output / name)
        result["windows"][name] = {"users": len(ordered_users), "channels": len(ordered_channels),
              "nonzeros": matrix.nnz, "raw_seconds": float(state.totals.sum()), "cutoff": cutoff,
              "profile_sha256": sha256(output / name / "profile.json"), "counts_sha256": sha256(output / name / "counts.npz")}
        counts[name].clear()
    after = database.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("Original SQLite changed")
    result["runtime_seconds"] = time.perf_counter() - started
    dump_json(output / "context-audit.json", result)
    return result


def schedule_features(database, source_audit, state, candidates, lower, upper, future=False):
    """Read only allowed schedule fields; future content attributes are never loaded."""
    connection = sqlite3.connect(Path(database).resolve().as_uri() + "?mode=ro", uri=True)
    connection.execute("PRAGMA query_only=ON")
    airings = defaultdict(list)
    try:
        for role in (["future"] if future else ["train", "valid"]):
            scale = source_audit["files"][role]["duration_scale_seconds"]
            for channel, start, duration, item in connection.execute(
                "SELECT channel,start,duration,item_id FROM schedule WHERE role=? AND duration>0 AND start<? AND start+duration*?>? ORDER BY channel,start,item_id,duration",
                (role, upper, scale, lower)):
                airings[channel].append((start, start + duration * scale, item))
    finally:
        connection.close()
    return availability_matrix(airings, candidates, state.channels, lower, upper, state.hour_width)
