"""
find_duplicate_questions.py

Find duplicate questions in the MMLU dataset, broken down by split and subject.

Match modes (--match)
---------------------
  question          Same question stem (default)
  question_options  Same stem + A/B/C/D option texts
  full              Same stem + options + label

Scope
-----
  Within each subject file, and also across subjects inside a split
  (same stem appearing under two subject names).

Output (under benchmarks/MMLU/results/)
---------------------------------------
  mmlu_duplicate_questions_detail.csv   one row per member of a duplicate group
  mmlu_duplicate_questions_summary.csv  counts per split / subject

Example
-------
  python find_duplicate_questions.py
  python find_duplicate_questions.py --match question_options --splits test,val
  python find_duplicate_questions.py --data_dir /content/data --normalize
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import os
import sys

import pandas as pd

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utilities import (  # noqa: E402
    add_data_dir_arg,
    assign_eval_ids,
    project_root,
    resolve_data_dir,
)

COLS = ["question", "A", "B", "C", "D", "label"]
SPLITS = ("test", "dev", "val")


def normalize_text(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n").strip()
    return " ".join(text.split())


def row_key(row: pd.Series, match: str, normalize: bool) -> str:
    def cell(name: str) -> str:
        raw = row.get(name, "")
        return normalize_text(raw) if normalize else (
            "" if raw is None or (isinstance(raw, float) and pd.isna(raw)) else str(raw)
        )

    if match == "question":
        payload = cell("question")
    elif match == "question_options":
        payload = "||".join(cell(c) for c in ["question", "A", "B", "C", "D"])
    elif match == "full":
        payload = "||".join(cell(c) for c in COLS)
    else:
        raise ValueError(f"Unknown match mode: {match}")
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def load_split(data_dir: str, split: str) -> pd.DataFrame:
    split_dir = os.path.join(data_dir, split)
    if not os.path.isdir(split_dir):
        return pd.DataFrame()

    frames = []
    for path in sorted(glob.glob(os.path.join(split_dir, "*.csv"))):
        base = os.path.basename(path)
        # anatomy_test.csv / anatomy_dev.csv / anatomy_val.csv
        subject = base
        for suffix in (f"_{split}.csv", ".csv"):
            if subject.endswith(suffix):
                subject = subject[: -len(suffix)]
                break

        df = pd.read_csv(path, header=None, names=COLS)
        df["split"] = split
        df["subject"] = subject
        df["source_file"] = base
        # 1-based row index within the subject CSV (excluding header; these files have none)
        df["row_in_file"] = range(1, len(df) + 1)
        df["row_no"] = df["row_in_file"]
        df = assign_eval_ids(df, subject=subject)
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def find_duplicates(df: pd.DataFrame, match: str, normalize: bool, within: str) -> pd.DataFrame:
    """
    within:
      subject  — duplicate keys must share split+subject
      split    — duplicate keys must share split (may span subjects)
    """
    if df.empty:
        return pd.DataFrame()

    work = df.copy()
    work["dup_key"] = work.apply(lambda r: row_key(r, match, normalize), axis=1)

    if within == "subject":
        group_cols = ["split", "subject", "dup_key"]
    elif within == "split":
        group_cols = ["split", "dup_key"]
    else:
        raise ValueError(f"Unknown within mode: {within}")

    counts = work.groupby(group_cols).size().rename("group_size").reset_index()
    work = work.merge(counts, on=group_cols, how="left")
    dups = work[work["group_size"] > 1].copy()
    if dups.empty:
        return dups

    dups["scope"] = within
    dups["match"] = match
    # Stable group id for reading the detail CSV
    dups["group_id"] = dups[group_cols].astype(str).agg("|".join, axis=1).map(
        lambda s: hashlib.md5(s.encode("utf-8")).hexdigest()[:10]
    )
    return dups.sort_values(
        ["split", "subject", "group_size", "dup_key", "row_in_file"],
        ascending=[True, True, False, True, True],
    )


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame(
            columns=[
                "scope",
                "split",
                "subject",
                "duplicate_groups",
                "duplicate_rows",
                "max_group_size",
            ]
        )

    rows = []
    sub = detail[detail["scope"] == "subject"]
    if not sub.empty:
        for (split, subject), grp in sub.groupby(["split", "subject"]):
            rows.append(
                {
                    "scope": "subject",
                    "split": split,
                    "subject": subject,
                    "duplicate_groups": int(grp["group_id"].nunique()),
                    "duplicate_rows": int(len(grp)),
                    "max_group_size": int(grp["group_size"].max()),
                }
            )
        for split, grp in sub.groupby("split"):
            rows.append(
                {
                    "scope": "subject",
                    "split": split,
                    "subject": "__ALL__",
                    "duplicate_groups": int(grp["group_id"].nunique()),
                    "duplicate_rows": int(len(grp)),
                    "max_group_size": int(grp["group_size"].max()),
                }
            )

    cross = detail[detail["scope"] == "cross_subject"]
    if not cross.empty:
        for split, grp in cross.groupby("split"):
            rows.append(
                {
                    "scope": "cross_subject",
                    "split": split,
                    "subject": "__CROSS_SUBJECT__",
                    "duplicate_groups": int(grp["group_id"].nunique()),
                    "duplicate_rows": int(len(grp)),
                    "max_group_size": int(grp["group_size"].max()),
                }
            )

    return pd.DataFrame(rows).sort_values(
        ["scope", "split", "duplicate_groups"], ascending=[True, True, False]
    )


def main():
    parser = argparse.ArgumentParser(
        description="Find duplicate MMLU questions per split and per subject."
    )
    add_data_dir_arg(parser)
    parser.add_argument(
        "--splits",
        type=str,
        default="test,dev,val",
        help="Comma-separated splits to scan (default: test,dev,val)",
    )
    parser.add_argument(
        "--match",
        choices=["question", "question_options", "full"],
        default="question",
        help="What must match to count as a duplicate (default: question)",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize whitespace before comparing (collapse runs of space/newlines)",
    )
    parser.add_argument(
        "--subjects",
        type=str,
        default="all",
        help="Comma-separated subjects, or 'all' (default: all)",
    )
    args = parser.parse_args()

    data_dir = resolve_data_dir(args.data_dir, from_file=__file__)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    print(f"Using data_dir: {data_dir}")
    print(f"Splits: {splits} | match={args.match} | normalize={args.normalize}")

    frames = []
    for split in splits:
        df = load_split(data_dir, split)
        if df.empty:
            print(f"  [{split}] no CSVs found — skipped")
            continue
        if args.subjects.strip().lower() != "all":
            wanted = {s.strip() for s in args.subjects.split(",") if s.strip()}
            df = df[df["subject"].isin(wanted)]
        print(f"  [{split}] {len(df)} questions across {df['subject'].nunique()} subjects")
        frames.append(df)

    if not frames:
        print("Nothing to scan.")
        return

    all_df = pd.concat(frames, ignore_index=True)

    within_subject = find_duplicates(
        all_df, match=args.match, normalize=args.normalize, within="subject"
    )
    within_split = find_duplicates(
        all_df, match=args.match, normalize=args.normalize, within="split"
    )

    # For split-scope detail, keep only groups that span multiple subjects
    # (same-subject dups are already covered by within_subject).
    if not within_split.empty:
        n_subj = within_split.groupby("group_id")["subject"].transform("nunique")
        within_split = within_split[n_subj > 1].copy()
        within_split["scope"] = "cross_subject"

    detail = pd.concat([within_subject, within_split], ignore_index=True)
    summary = summarize(detail)

    out_dir = os.path.join(project_root(__file__), "benchmarks", "MMLU", "results")
    os.makedirs(out_dir, exist_ok=True)
    detail_path = os.path.join(out_dir, "mmlu_duplicate_questions_detail.csv")
    summary_path = os.path.join(out_dir, "mmlu_duplicate_questions_summary.csv")

    detail_cols = [
        "scope",
        "split",
        "subject",
        "group_id",
        "group_size",
        "row_in_file",
        "question_id",
        "row_id",
        "source_file",
        "question",
        "A",
        "B",
        "C",
        "D",
        "label",
        "match",
        "dup_key",
    ]
    if detail.empty:
        pd.DataFrame(columns=detail_cols).to_csv(detail_path, index=False)
    else:
        detail[detail_cols].to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("\n=== SUMMARY (within-subject duplicates) ===")
    sub_sum = summary[summary["scope"] == "subject"]
    if sub_sum.empty:
        print("No within-subject duplicate questions found.")
    else:
        # Show per-split ALL rows + subjects that have dups
        show = sub_sum.copy()
        print(show.to_string(index=False))

    print("\n=== CROSS-SUBJECT (same question text in multiple subjects, same split) ===")
    cross_sum = summary[summary["scope"] == "cross_subject"]
    if cross_sum.empty:
        print("None found.")
    else:
        print(cross_sum.to_string(index=False))

    print(f"\nDetail:  {detail_path}  ({len(detail)} rows)")
    print(f"Summary: {summary_path}  ({len(summary)} rows)")


if __name__ == "__main__":
    main()
