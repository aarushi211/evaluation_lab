"""
find_duplicate_questions.py

Find duplicate questions in the MMLU dataset, broken down by split and subject.

Match modes (--match)
---------------------
  question          Same stem only — broad *screening* (shared wording, maybe
                    different options). Diagnostic, not a cleaning decision.
  question_options  Same stem + A/B/C/D — primary definition of a repeated
                    *evaluation item* (default). Use this for cleaning.
  full              Same stem + options + label — exact-record diagnostic.
                    Do NOT use as the main cleaner: label disagreements on the
                    same item are hidden by this mode (see label-conflict CSV).
  all               Run question, question_options, and full (separate outputs).

Whitespace
----------
  Normalized by default (--normalize). Pass --no-normalize for byte-exact text.

Interpretation
--------------
  Same stem only              → potential / soft duplicate
  Same stem + options         → strong duplicate (same item)
  Same stem + options + label → exact duplicate record
  Same stem + options, ≠ label → inconsistent duplicate (flagged separately)

Output (under benchmarks/MMLU/results/)
---------------------------------------
  mmlu_duplicate_questions_detail[_{match}].csv
  mmlu_duplicate_questions_summary[_{match}].csv
  mmlu_duplicate_label_conflicts.csv
      (same question+options, conflicting ground-truth labels)

Example
-------
  # Primary cleaning scan (default)
  python find_duplicate_questions.py

  # Broad screening only
  python find_duplicate_questions.py --match question

  # All three definitions + conflicts
  python find_duplicate_questions.py --match all

  python find_duplicate_questions.py --splits test --no-normalize
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
MATCH_MODES = ("question", "question_options", "full")
DETAIL_COLS = [
    "scope",
    "split",
    "subject",
    "group_id",
    "group_size",
    "excess_rows",
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


def normalize_text(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n").strip()
    return " ".join(text.split())


def cell_text(row: pd.Series, name: str, normalize: bool) -> str:
    raw = row.get(name, "")
    if normalize:
        return normalize_text(raw)
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return ""
    return str(raw)


def row_key(row: pd.Series, match: str, normalize: bool) -> str:
    if match == "question":
        payload = cell_text(row, "question", normalize)
    elif match == "question_options":
        payload = "||".join(cell_text(row, c, normalize) for c in ["question", "A", "B", "C", "D"])
    elif match == "full":
        payload = "||".join(cell_text(row, c, normalize) for c in COLS)
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
        subject = base
        for suffix in (f"_{split}.csv", ".csv"):
            if subject.endswith(suffix):
                subject = subject[: -len(suffix)]
                break

        df = pd.read_csv(path, header=None, names=COLS)
        df["split"] = split
        df["subject"] = subject
        df["source_file"] = base
        df["row_in_file"] = range(1, len(df) + 1)
        df["row_no"] = df["row_in_file"]
        df = assign_eval_ids(df, subject=subject)
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def find_duplicates(
    df: pd.DataFrame, match: str, normalize: bool, within: str
) -> pd.DataFrame:
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

    dups["excess_rows"] = dups["group_size"] - 1
    dups["scope"] = within
    dups["match"] = match
    dups["group_id"] = dups[group_cols].astype(str).agg("|".join, axis=1).map(
        lambda s: hashlib.md5(s.encode("utf-8")).hexdigest()[:10]
    )
    return dups.sort_values(
        ["split", "subject", "group_size", "dup_key", "row_in_file"],
        ascending=[True, True, False, True, True],
    )


def find_label_conflicts(df: pd.DataFrame, normalize: bool) -> pd.DataFrame:
    """
    Same question + options (item key), but more than one distinct label.

    These are inconsistent duplicates — more serious than ordinary repetition.
    Hidden if you only scan with --match full.
    """
    if df.empty:
        return pd.DataFrame()

    work = df.copy()
    work["item_key"] = work.apply(
        lambda r: row_key(r, "question_options", normalize), axis=1
    )
    work["label_norm"] = work.apply(lambda r: cell_text(r, "label", normalize), axis=1)

    # Per split + item (cross-subject conflicts included)
    stats = (
        work.groupby(["split", "item_key"])
        .agg(
            n_rows=("label_norm", "size"),
            n_labels=("label_norm", "nunique"),
            labels=("label_norm", lambda s: "|".join(sorted(set(s)))),
            n_subjects=("subject", "nunique"),
            subjects=("subject", lambda s: "|".join(sorted(set(s)))),
        )
        .reset_index()
    )
    bad_keys = stats[stats["n_labels"] > 1][["split", "item_key", "n_rows", "n_labels", "labels", "n_subjects", "subjects"]]
    if bad_keys.empty:
        return pd.DataFrame()

    detail = work.merge(bad_keys, on=["split", "item_key"], how="inner")
    detail["group_id"] = detail[["split", "item_key"]].astype(str).agg("|".join, axis=1).map(
        lambda s: hashlib.md5(s.encode("utf-8")).hexdigest()[:10]
    )
    detail["conflict_type"] = "label_mismatch"
    detail["excess_rows"] = detail["n_rows"] - 1
    return detail.sort_values(
        ["split", "n_labels", "item_key", "subject", "row_in_file"],
        ascending=[True, False, True, True, True],
    )


def _group_excess(grp: pd.DataFrame) -> int:
    """Sum of (group_size - 1) over unique groups (= duplicate_rows - n_groups)."""
    per_group = grp.drop_duplicates("group_id")["group_size"]
    return int((per_group - 1).sum())


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    empty_cols = [
        "scope",
        "split",
        "subject",
        "duplicate_groups",
        "duplicate_rows",
        "excess_rows",
        "max_group_size",
    ]
    if detail.empty:
        return pd.DataFrame(columns=empty_cols)

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
                    "excess_rows": _group_excess(grp),
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
                    "excess_rows": _group_excess(grp),
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
                    "excess_rows": _group_excess(grp),
                    "max_group_size": int(grp["group_size"].max()),
                }
            )

    return pd.DataFrame(rows).sort_values(
        ["scope", "split", "duplicate_groups"], ascending=[True, True, False]
    )


def build_detail(all_df: pd.DataFrame, match: str, normalize: bool) -> pd.DataFrame:
    within_subject = find_duplicates(
        all_df, match=match, normalize=normalize, within="subject"
    )
    within_split = find_duplicates(
        all_df, match=match, normalize=normalize, within="split"
    )
    if not within_split.empty:
        n_subj = within_split.groupby("group_id")["subject"].transform("nunique")
        within_split = within_split[n_subj > 1].copy()
        within_split["scope"] = "cross_subject"

    if within_subject.empty and within_split.empty:
        return pd.DataFrame(columns=DETAIL_COLS)
    return pd.concat([within_subject, within_split], ignore_index=True)


def write_outputs(
    detail: pd.DataFrame,
    summary: pd.DataFrame,
    out_dir: str,
    match: str,
) -> tuple[str, str]:
    suffix = f"_{match}"
    detail_path = os.path.join(out_dir, f"mmlu_duplicate_questions_detail{suffix}.csv")
    summary_path = os.path.join(out_dir, f"mmlu_duplicate_questions_summary{suffix}.csv")

    if detail.empty:
        pd.DataFrame(columns=DETAIL_COLS).to_csv(detail_path, index=False)
    else:
        cols = [c for c in DETAIL_COLS if c in detail.columns]
        detail[cols].to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    return detail_path, summary_path


def print_summary(match: str, summary: pd.DataFrame, detail: pd.DataFrame) -> None:
    print(f"\n=== MATCH={match} — WITHIN-SUBJECT ===")
    sub_sum = summary[summary["scope"] == "subject"] if not summary.empty else summary
    if sub_sum.empty:
        print("No within-subject duplicates.")
    else:
        print(sub_sum.to_string(index=False))

    print(f"\n=== MATCH={match} — CROSS-SUBJECT ===")
    cross_sum = (
        summary[summary["scope"] == "cross_subject"] if not summary.empty else summary
    )
    if cross_sum.empty:
        print("None found.")
    else:
        print(cross_sum.to_string(index=False))

    if not detail.empty:
        n_groups = detail["group_id"].nunique()
        excess = _group_excess(detail)
        print(
            f"\n[{match}] {len(detail)} duplicate rows in {n_groups} groups "
            f"-> {excess} excess rows (copies beyond the first)."
        )


def run_one_match(
    all_df: pd.DataFrame, match: str, normalize: bool, out_dir: str
) -> None:
    detail = build_detail(all_df, match=match, normalize=normalize)
    summary = summarize(detail)
    detail_path, summary_path = write_outputs(detail, summary, out_dir, match)
    print_summary(match, summary, detail)
    print(f"Detail:  {detail_path}")
    print(f"Summary: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Find duplicate MMLU questions per split/subject. "
            "Default: --match question_options with whitespace normalization "
            "(primary cleaning definition). Use --match question for broad screening."
        )
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
        choices=[*MATCH_MODES, "all"],
        default="question_options",
        help=(
            "Duplicate definition: question (screening), question_options "
            "(primary / default), full (exact record), or all"
        ),
    )
    parser.add_argument(
        "--normalize",
        dest="normalize",
        action="store_true",
        default=True,
        help="Normalize whitespace before comparing (default: on)",
    )
    parser.add_argument(
        "--no-normalize",
        dest="normalize",
        action="store_false",
        help="Disable whitespace normalization (byte-exact text)",
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
    if args.match == "question":
        print(
            "Note: --match question is a broad screening mode. "
            "Prefer question_options for cleaning decisions."
        )
    if args.match == "full":
        print(
            "Note: --match full includes the label. Label conflicts on the same "
            "item are hidden here — see mmlu_duplicate_label_conflicts.csv."
        )

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
    out_dir = os.path.join(project_root(__file__), "benchmarks", "MMLU", "results")
    os.makedirs(out_dir, exist_ok=True)

    modes = list(MATCH_MODES) if args.match == "all" else [args.match]
    for match in modes:
        run_one_match(all_df, match=match, normalize=args.normalize, out_dir=out_dir)

    # Always emit label-conflict report (item-level; independent of --match)
    conflicts = find_label_conflicts(all_df, normalize=args.normalize)
    conflict_path = os.path.join(out_dir, "mmlu_duplicate_label_conflicts.csv")
    conflict_cols = [
        "conflict_type",
        "split",
        "group_id",
        "subject",
        "subjects",
        "n_subjects",
        "n_rows",
        "n_labels",
        "labels",
        "excess_rows",
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
        "item_key",
    ]
    if conflicts.empty:
        pd.DataFrame(columns=conflict_cols).to_csv(conflict_path, index=False)
        print("\n=== LABEL CONFLICTS (same question+options, different labels) ===")
        print("None found.")
    else:
        cols = [c for c in conflict_cols if c in conflicts.columns]
        conflicts[cols].to_csv(conflict_path, index=False)
        n_groups = conflicts["group_id"].nunique()
        print("\n=== LABEL CONFLICTS (same question+options, different labels) ===")
        print(
            f"{len(conflicts)} rows in {n_groups} inconsistent item groups "
            f"across splits {sorted(conflicts['split'].unique())}."
        )
        brief = (
            conflicts.drop_duplicates("group_id")[
                ["split", "subjects", "n_labels", "labels", "n_rows"]
            ]
            .sort_values(["split", "n_rows"], ascending=[True, False])
        )
        print(brief.to_string(index=False))
    print(f"Conflicts: {conflict_path}")


if __name__ == "__main__":
    main()
