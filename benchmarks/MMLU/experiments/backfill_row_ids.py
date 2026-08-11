"""
backfill_row_ids.py

Add row_id (and keep question_id) on an existing results CSV without re-running.

Matching
--------
  evaluate_mmlu-style files (subject + question_no):
      row_id = MD5(subject::question_no) after checking it matches the source CSV.

  Files with question text:
      join to the source row via subject + question_id.

--expand
--------
  Append source rows that were skipped as duplicates (same question_id already
  evaluated). Copies predicted / correct / raw_output from the sibling result
  so you do not need new API calls.

Example
-------
  python backfill_row_ids.py \\
      --results ../results/openai_gpt-5-nano_all_no_shuffle_0shot_lim100000.csv \\
      --expand
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import shutil
import sys

import pandas as pd

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utilities import (  # noqa: E402
    add_data_dir_arg,
    assign_eval_ids,
    resolve_data_dir,
)


def load_source_index(data_dir: str, split: str = "test") -> pd.DataFrame:
    frames = []
    for path in sorted(glob.glob(os.path.join(data_dir, split, "*.csv"))):
        subject = os.path.basename(path).replace(f"_{split}.csv", "")
        df = pd.read_csv(
            path, header=None, names=["question", "A", "B", "C", "D", "label"]
        )
        df["subject"] = subject
        df["row_no"] = df.index + 1
        df = assign_eval_ids(df, subject=subject)
        frames.append(
            df[
                [
                    "subject",
                    "row_no",
                    "question_id",
                    "row_id",
                    "question",
                    "label",
                ]
            ]
        )
    if not frames:
        raise FileNotFoundError(f"No CSVs under {os.path.join(data_dir, split)}")
    return pd.concat(frames, ignore_index=True)


def _copy_prediction(template: pd.Series, source_row: pd.Series) -> dict:
    row = template.to_dict()
    row["question_id"] = source_row["question_id"]
    row["row_id"] = source_row["row_id"]
    if "question_no" in row:
        row["question_no"] = int(source_row["row_no"])
    if "subject" in row:
        row["subject"] = source_row["subject"]
    if "ground_truth" in row and pd.notna(source_row.get("label")):
        row["ground_truth"] = str(source_row["label"]).strip()
        # Recompute correct if we copied a prediction and GT might differ
        # (duplicate stems usually share the same label).
        if "predicted" in row and pd.notna(row.get("predicted")):
            pred = str(row["predicted"]).strip()
            gt = str(row["ground_truth"]).strip()
            if "correct" in row:
                row["correct"] = str(pred == gt)
    if "question" in row:
        row["question"] = source_row["question"]
    return row


def backfill(results: pd.DataFrame, source: pd.DataFrame, expand: bool) -> pd.DataFrame:
    out = results.copy()
    has_qno = "question_no" in out.columns and "subject" in out.columns

    if has_qno:
        out["question_no"] = pd.to_numeric(out["question_no"], errors="coerce")
        merged = out.merge(
            source[["subject", "row_no", "question_id", "row_id"]].rename(
                columns={"question_id": "qid_src", "row_id": "row_id_src"}
            ),
            left_on=["subject", "question_no"],
            right_on=["subject", "row_no"],
            how="left",
        )
        mismatch = merged["row_id_src"].notna() & (
            merged["question_id"].astype(str) != merged["qid_src"].astype(str)
        )
        if int(mismatch.sum()):
            raise ValueError(
                f"{int(mismatch.sum())} rows: question_no does not match "
                "source question_id. Refusing to backfill."
            )
        out["row_id"] = merged["row_id_src"]
        matched = int(out["row_id"].notna().sum())
        print(f"Matched {matched}/{len(out)} result rows via subject + question_no.")
    else:
        # Greedy match on subject + question_id (first unused source row).
        src_unused = source.copy()
        src_unused["_taken"] = False
        row_ids = []
        for _, row in out.iterrows():
            subj = row.get("subject")
            qid = str(row["question_id"])
            hit = src_unused[
                (~src_unused["_taken"])
                & (src_unused["subject"] == subj)
                & (src_unused["question_id"] == qid)
            ]
            if hit.empty:
                row_ids.append(None)
            else:
                idx = hit.index[0]
                row_ids.append(src_unused.at[idx, "row_id"])
                src_unused.at[idx, "_taken"] = True
        out["row_id"] = row_ids
        print(
            f"Matched {out['row_id'].notna().sum()}/{len(out)} result rows "
            "via subject + question_id."
        )

    def _order_cols(df: pd.DataFrame) -> pd.DataFrame:
        cols = list(df.columns)
        if "row_id" in cols:
            cols.remove("row_id")
            qidx = cols.index("question_id") + 1 if "question_id" in cols else 1
            cols.insert(qidx, "row_id")
        return df[cols]

    if not expand:
        return _order_cols(out)

    used = set()
    if has_qno:
        used = set(
            zip(out["subject"].astype(str), out["question_no"].astype(int))
        )
        missing_src = source[
            ~source.apply(lambda r: (str(r["subject"]), int(r["row_no"])) in used, axis=1)
        ]
    else:
        used_ids = set(out["row_id"].dropna().astype(str))
        missing_src = source[~source["row_id"].astype(str).isin(used_ids)]

    # Only expand rows whose question_id was already evaluated (true skipped dups).
    have_qids = set(out["question_id"].astype(str))
    missing_src = missing_src[missing_src["question_id"].astype(str).isin(have_qids)]
    if missing_src.empty:
        print("No skipped duplicate source rows to expand.")
        return out

    extras = []
    for _, src_row in missing_src.iterrows():
        sib = out[out["question_id"].astype(str) == str(src_row["question_id"])]
        if sib.empty:
            continue
        extras.append(_copy_prediction(sib.iloc[0], src_row))

    extra_df = pd.DataFrame(extras)
    print(
        f"Expanding {len(extra_df)} skipped duplicate rows "
        f"(copied predictions from the matching question_id)."
    )
    combined = pd.concat([out, extra_df], ignore_index=True)
    return _order_cols(combined)


def main():
    parser = argparse.ArgumentParser(
        description="Backfill row_id onto an existing results CSV (no re-run)."
    )
    add_data_dir_arg(parser)
    parser.add_argument("--results", required=True, help="Results CSV to update")
    parser.add_argument(
        "--out",
        default=None,
        help="Output path (default: overwrite --results after a .bak sidecar)",
    )
    parser.add_argument(
        "--expand",
        action="store_true",
        help="Append skipped duplicate source rows, copying the sibling prediction",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split to index (default: test)",
    )
    args = parser.parse_args()

    data_dir = resolve_data_dir(args.data_dir, from_file=__file__)
    print(f"Using data_dir: {data_dir}")
    source = load_source_index(data_dir, args.split)
    print(f"Source {args.split}: {len(source)} rows, {source['question_id'].nunique()} unique question_ids")

    results = pd.read_csv(args.results, dtype=str)
    if "question_id" not in results.columns:
        raise ValueError(f"{args.results} has no question_id column")

    updated = backfill(results, source, expand=args.expand)
    missing_ids = updated["row_id"].isna().sum() if "row_id" in updated.columns else len(updated)
    print(f"Rows still without row_id: {int(missing_ids)}")

    out_path = args.out or args.results
    if out_path == args.results:
        bak = args.results + ".bak_before_row_id"
        if not os.path.exists(bak):
            shutil.copy2(args.results, bak)
            print(f"Backup -> {bak}")

    updated.to_csv(out_path, index=False, quoting=csv.QUOTE_ALL)
    print(f"Wrote {len(updated)} rows -> {out_path}")
    print(
        f"unique question_id={updated['question_id'].nunique()}  "
        f"unique row_id={updated['row_id'].nunique(dropna=True)}"
    )


if __name__ == "__main__":
    main()
