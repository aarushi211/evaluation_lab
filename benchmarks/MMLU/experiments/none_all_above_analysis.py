"""
none_all_above_analysis.py

Check: How often is "none of the above" / "all of the above" (or similar
catch-all options) the correct answer, and are these options a giveaway
by their mere presence?

Pure dataset-artifact check — no model calls needed. Two things measured:
1. How often a catch-all option is present at all, per split/category.
2. When present, how often it's actually the correct answer (compared
   to the 25% chance baseline).

If catch-all options are correct far more/less than 25% of the time
whenever they appear, that's an exploitable prior: a model (or human)
could learn "when in doubt, pick the 'all of the above' option" (or
avoid it) without understanding the question at all.
"""

import os
import re
import glob
import argparse
import pandas as pd

from subject_bias_analysis import get_category

LABELS = ['A', 'B', 'C', 'D']

CATCHALL_PATTERNS = [
    r'^\s*all of the above\s*\.?\s*$',
    r'^\s*none of the above\s*\.?\s*$',
    r'^\s*both .* and .*\s*$',
    r'^\s*all of these\s*\.?\s*$',
    r'^\s*none of these\s*\.?\s*$',
]
CATCHALL_REGEX = re.compile('|'.join(CATCHALL_PATTERNS), flags=re.IGNORECASE)


def is_catchall(option_text: str) -> bool:
    return bool(CATCHALL_REGEX.match(str(option_text).strip()))


def load_all(data_dir: str, split: str) -> pd.DataFrame:
    split_dir = os.path.join(data_dir, split)
    csv_files = glob.glob(os.path.join(split_dir, "*.csv"))
    rows = []
    for f in csv_files:
        subject = os.path.basename(f).replace(f"_{split}.csv", "")
        try:
            df = pd.read_csv(f, header=None, names=['question', 'A', 'B', 'C', 'D', 'label'])
            df['subject'] = subject
            df['category'] = get_category(subject)
            rows.append(df)
        except Exception as e:
            print(f"Skipping {f}: {e}")
    if not rows:
        return pd.DataFrame()
    combined = pd.concat(rows, ignore_index=True)
    combined['label'] = combined['label'].astype(str).str.strip()
    return combined[combined['label'].isin(LABELS)]


def annotate_catchall(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for l in LABELS:
        df[f'{l}_is_catchall'] = df[l].apply(is_catchall)

    df['has_catchall'] = df[[f'{l}_is_catchall' for l in LABELS]].any(axis=1)

    def catchall_letter(row):
        for l in LABELS:
            if row[f'{l}_is_catchall']:
                return l
        return None

    df['catchall_letter'] = df.apply(catchall_letter, axis=1)
    df['catchall_is_correct'] = df.apply(
        lambda r: (r['catchall_letter'] == r['label']) if r['has_catchall'] else None, axis=1
    )
    return df


def report(df: pd.DataFrame, label: str):
    total = len(df)
    with_catchall = df[df['has_catchall']]
    n_catchall = len(with_catchall)
    pct_present = (n_catchall / total * 100) if total else 0

    print(f"\n=== {label} (n={total}) ===")
    print(f"  Questions containing a catch-all option: {n_catchall} ({pct_present:.2f}%)")

    if n_catchall > 0:
        correct_when_present = int(with_catchall['catchall_is_correct'].astype(bool).sum())
        pct_correct = correct_when_present / n_catchall * 100
        diff = pct_correct - 25.0
        direction = "MORE" if diff > 0 else "LESS"
        print(f"  Of those, catch-all option is correct: {correct_when_present}/{n_catchall} "
              f"({pct_correct:.2f}%) — {direction} often than the 25% chance baseline "
              f"(diff={diff:+.2f}pp)")


def run(args):
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.abspath(os.path.join(_script_dir, '..', '..', '..'))
    data_dir = os.path.join(_project_root, 'datasets', 'MMLU', 'data', 'data')

    df = load_all(data_dir, args.split)
    if df.empty:
        print(f"No data found for split '{args.split}'.")
        return

    df = annotate_catchall(df)

    report(df, f"OVERALL ({args.split} split)")

    print("\n\n########## BY ACADEMIC CATEGORY ##########")
    for cat in sorted(df['category'].unique()):
        report(df[df['category'] == cat], cat)

    if args.list_examples:
        print("\n\n########## SAMPLE QUESTIONS WITH CATCH-ALL OPTIONS ##########")
        examples = df[df['has_catchall']].head(args.list_examples)
        for _, row in examples.iterrows():
            print(f"  [{row['subject']}] Catchall={row['catchall_letter']} | "
                  f"GroundTruth={row['label']} | Q: {str(row['question'])[:80]}...")

    out_dir = os.path.join(_script_dir, '..', 'analysis')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"none_all_above_{args.split}.csv")
    cols = ['subject', 'category', 'question', 'label', 'has_catchall', 'catchall_letter', 'catchall_is_correct']
    df[cols].to_csv(out_path, index=False)
    print(f"\nSaved detailed results to: {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Check how often 'all/none of the above' style options appear and are correct (no API calls)"
    )
    parser.add_argument('--split', type=str, default='test', choices=['test', 'val', 'dev'])
    parser.add_argument('--list_examples', type=int, default=0,
                         help="Print this many example questions containing a catch-all option")
    run(parser.parse_args())