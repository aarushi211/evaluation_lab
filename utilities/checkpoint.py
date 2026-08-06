"""Append-only result checkpointing for long evaluation runs."""

from __future__ import annotations

import csv
import hashlib
import os
from typing import Dict, Iterable, Optional, Set


def make_question_id(subject: str, question: str) -> str:
    """Stable content-based ID so resuming works even if sampling order changes."""
    return hashlib.md5(f"{subject}::{question}".encode("utf-8")).hexdigest()[:12]


def load_processed_ids(out_path: str, id_column: str = "question_id") -> Set[str]:
    if not os.path.exists(out_path):
        return set()
    try:
        import pandas as pd

        existing = pd.read_csv(out_path)
        if id_column not in existing.columns:
            return set()
        return set(existing[id_column].astype(str))
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
    """
    keys = list(fieldnames) if fieldnames is not None else list(row.keys())
    file_exists = os.path.exists(out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())
