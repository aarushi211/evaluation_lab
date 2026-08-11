"""
checkpoint.py

Append-only CSV helpers so long evaluation runs can resume after interruption.

IDs
---
  question_id = MD5(subject::question)[:12]
      Content-based: the same stem in the same subject always shares this ID
      (including exact duplicate questions).
  row_id = MD5(subject::row_number)[:12]
      Instance-based: unique per CSV row, so duplicate stems still run
      separately. Resume skips completed row_ids.

Functions
---------
  make_question_id(subject, question)
  make_row_id(subject, row_number)
  assign_eval_ids(df, subject=None, subject_col="subject", row_col="row_no")
  load_processed_ids(out_path, id_column=None)
      Prefers row_id; falls back to question_id on older result files.
  append_result_row(out_path, row, fieldnames=None)
"""

from __future__ import annotations

import csv
import hashlib
import os
import threading
from typing import Dict, Iterable, Optional, Set

import pandas as pd

# Serialize appends so --workers > 1 cannot interleave CSV rows/headers.
_CSV_LOCK = threading.Lock()


def make_question_id(subject: str, question: str) -> str:
    """Stable content-based ID: same subject + question stem → same ID."""
    return hashlib.md5(f"{subject}::{question}".encode("utf-8")).hexdigest()[:12]


def make_row_id(subject: str, row_number: int) -> str:
    """Stable instance ID: same subject + original CSV row number → same ID."""
    return hashlib.md5(f"{subject}::{int(row_number)}".encode("utf-8")).hexdigest()[:12]


def assign_eval_ids(
    df: pd.DataFrame,
    *,
    subject: Optional[str] = None,
    subject_col: str = "subject",
    row_col: str = "row_no",
) -> pd.DataFrame:
    """
    Add question_id and row_id. Requires a 1-based original-file row number
    in `row_col` (set before any sampling / concat that drops the index).
    """
    out = df.copy()
    if subject is not None:
        subj = lambda _: subject  # noqa: E731
    else:
        subj = lambda r: r[subject_col]  # noqa: E731
    out["question_id"] = out.apply(
        lambda r: make_question_id(subj(r), r["question"]), axis=1
    )
    out["row_id"] = out.apply(
        lambda r: make_row_id(subj(r), r[row_col]), axis=1
    )
    return out


def filter_unprocessed(df: pd.DataFrame, out_path: str) -> pd.DataFrame:
    """
    Drop rows already present in the results CSV.

    Uses row_id when the existing file has that column (duplicate stems are
    kept as separate work items). Falls back to question_id for older files.
    """
    if df.empty or not os.path.exists(out_path):
        return df
    processed = load_processed_ids(out_path)
    if not processed:
        return df
    try:
        existing_cols = set(pd.read_csv(out_path, dtype=str, nrows=0).columns)
    except Exception:
        existing_cols = set()
    if "row_id" in existing_cols and "row_id" in df.columns:
        return df[~df["row_id"].astype(str).isin(processed)]
    if "question_id" in df.columns:
        return df[~df["question_id"].astype(str).isin(processed)]
    return df


def load_processed_ids(out_path: str, id_column: Optional[str] = None) -> Set[str]:
    """
    IDs already written to an existing results CSV.

    Resume key is row_id (so duplicate questions are not skipped). Older files
    that only have question_id still resume on that column.
    """
    if not os.path.exists(out_path):
        return set()
    try:
        existing = pd.read_csv(out_path, dtype=str)
        col = id_column
        if col is None:
            if "row_id" in existing.columns:
                col = "row_id"
            elif "question_id" in existing.columns:
                col = "question_id"
            else:
                return set()
        if col not in existing.columns:
            return set()
        return set(existing[col].dropna().astype(str))
    except Exception:
        return set()


def append_result_row(
    out_path: str,
    row: Dict,
    fieldnames: Optional[Iterable[str]] = None,
) -> None:
    """
    Append one result row to a CSV, creating the header on first write.
    Flushes + fsyncs so an interrupted run does not lose the last row.
    Thread-safe for concurrent workers.
    """
    keys = list(fieldnames) if fieldnames is not None else list(row.keys())
    with _CSV_LOCK:
        file_exists = os.path.exists(out_path) and os.path.getsize(out_path) > 0
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        existing_fields = None
        if file_exists:
            with open(out_path, "r", newline="", encoding="utf-8") as rf:
                reader = csv.reader(rf)
                existing_fields = next(reader, None)
        write_fields = existing_fields if existing_fields else keys
        with open(out_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=write_fields,
                extrasaction="ignore",
                quoting=csv.QUOTE_ALL,
            )
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
            f.flush()
            os.fsync(f.fileno())
