"""Strict template-preserving SWEET.TV submission writing and validation."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
import re
from typing import Mapping, Sequence


_FIVE_IDS = re.compile(r"[0-9]+(?: [0-9]+){4}\Z", flags=re.ASCII)
_NULL_STRINGS = {"", "null", "none", "nan", "<na>"}


def _read_csv(path: str | Path) -> tuple[list[str], list[list[str]], csv.Dialect, bool]:
    raw = Path(path).read_bytes()
    has_bom = raw.startswith(b"\xef\xbb\xbf")
    text = raw.decode("utf-8-sig")
    try:
        dialect = csv.Sniffer().sniff(text[:65536], delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        # Whitespace in a user ID is data, even if Sniffer suggests otherwise.
        reader = csv.reader(stream, dialect=dialect, skipinitialspace=False, strict=True)
        try:
            header = next(reader)
        except StopIteration as error:
            raise ValueError(f"CSV is empty: {path}") from error
        rows = list(reader)
    if not header or len(set(header)) != len(header):
        raise ValueError(f"CSV has empty or duplicate headers: {path}")
    if any(len(row) != len(header) for row in rows):
        raise ValueError(f"CSV row width differs from its header: {path}")
    return header, rows, dialect, has_bom


def _columns(header: list[str]) -> tuple[int, int]:
    if "user_id" not in header:
        raise ValueError("The template must contain the exact user_id column")
    user_column = header.index("user_id")
    prediction_columns = [index for index in range(len(header)) if index != user_column]
    if len(prediction_columns) != 1:
        raise ValueError("Ambiguous prediction column: expected one non-user field")
    return user_column, prediction_columns[0]


def _users(rows: list[list[str]], user_column: int, expected_users: int) -> list[str]:
    if len(rows) != expected_users:
        raise ValueError(f"Expected {expected_users} users, found {len(rows)}")
    users = [row[user_column] for row in rows]
    if any(user.strip().lower() in _NULL_STRINGS for user in users):
        raise ValueError("User IDs contain null values")
    if len(set(users)) != len(users):
        raise ValueError("User IDs contain duplicates")
    return users


def validate_submission(
    template_path: str | Path,
    submission_path: str | Path,
    candidates: set[int],
    expected_users: int = 1260,
) -> dict:
    """Validate headers, raw user strings/order, spacing and five eligible IDs.

    IDs must be positive decimal integer tokens, distinct within each row, and
    members of the supplied future-schedule candidate set.  No numeric parsing
    is applied to user IDs.  The returned SHA-256 identifies the exact CSV bytes
    which passed this check.
    """
    if expected_users <= 0:
        raise ValueError("expected_users must be positive")
    template_header, template_rows, _, _ = _read_csv(template_path)
    user_column, prediction_column = _columns(template_header)
    template_users = _users(template_rows, user_column, expected_users)
    header, rows, _, _ = _read_csv(submission_path)
    if header != template_header:
        raise ValueError("Submission headers differ from the original template")
    users = _users(rows, user_column, expected_users)
    if users != template_users:
        raise ValueError("Submission user IDs or row order differ from the template")
    candidate_ids = set(candidates)
    if any(type(item) is not int or item <= 0 for item in candidate_ids):
        raise ValueError("Candidate IDs must be positive integers")
    if len(candidate_ids) < 5:
        raise ValueError("At least five eligible candidates are required")
    distinct_recommended = set()
    for line, row in enumerate(rows, start=2):
        value = row[prediction_column]
        if not _FIVE_IDS.fullmatch(value):
            raise ValueError(f"Line {line}: require five integer IDs separated by single spaces")
        ids = [int(token) for token in value.split(" ")]
        if any(item <= 0 for item in ids):
            raise ValueError(f"Line {line}: programme IDs must be positive; zeros are forbidden")
        if len(set(ids)) != 5:
            raise ValueError(f"Line {line}: duplicate programme IDs")
        if not set(ids).issubset(candidate_ids):
            raise ValueError(f"Line {line}: programme outside candidate pool")
        distinct_recommended.update(ids)
    raw = Path(submission_path).read_bytes()
    return {
        "passed": True,
        "path": str(Path(submission_path)),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
        "rows": len(rows),
        "expected_users": expected_users,
        "headers": header,
        "user_column": header[user_column],
        "prediction_column": header[prediction_column],
        "recommendations_per_user": 5,
        "candidate_count": len(candidate_ids),
        "distinct_recommended_programmes": len(distinct_recommended),
        "template_sha256": hashlib.sha256(Path(template_path).read_bytes()).hexdigest(),
    }


def write_submission(
    template_path: str | Path,
    output_path: str | Path,
    recommendations: Mapping[str, Sequence[int]],
    candidates: set[int],
    expected_users: int = 1260,
) -> dict:
    """Fill a template in its original user order, then validate the saved file."""
    template = Path(template_path)
    destination = Path(output_path)
    if destination.resolve() == template.resolve():
        raise ValueError("The original submission template must not be overwritten")
    header, rows, dialect, has_bom = _read_csv(template)
    user_column, prediction_column = _columns(header)
    users = _users(rows, user_column, expected_users)
    missing = [user for user in users if user not in recommendations]
    if missing:
        raise ValueError(f"Missing recommendations for template users: {missing[:5]}")
    completed = []
    for user, row in zip(users, rows):
        ids = list(recommendations[user])
        if len(ids) != 5 or any(type(item) is not int or item <= 0 for item in ids):
            raise ValueError(f"User {user!r}: require exactly five positive integer IDs")
        if len(set(ids)) != 5 or not set(ids).issubset(candidates):
            raise ValueError(f"User {user!r}: require distinct IDs from the candidate pool")
        row = list(row)
        row[prediction_column] = " ".join(map(str, ids))
        completed.append(row)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8-sig" if has_bom else "utf-8", newline="") as stream:
        writer = csv.writer(stream, dialect=dialect)
        writer.writerow(header)
        writer.writerows(completed)
    return validate_submission(template, destination, candidates, expected_users)
