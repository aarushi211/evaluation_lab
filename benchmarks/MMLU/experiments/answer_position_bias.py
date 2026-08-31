"""
answer_position_bias.py

Check: Is MMLU's answer key itself skewed toward a particular letter?

This is a pure dataset-artifact check — no model calls needed. If the
correct answer is disproportionately B or C (say), a model (or a human)
could exploit that prior without reading the question at all.

Uses a chi-square goodness-of-fit test against the expected uniform
25%/25%/25%/25% distribution, both overall and per subject.

Output
------
  benchmarks/MMLU/results/answer_position_bias_{split}.csv

Arguments
---------
  --split         Which MMLU split to scan  {test, val, dev}  (default: test)
  --per_subject   Also print a per-subject skew ranking       (flag)
  --data_dir      Folder containing test/val/dev — Colab-friendly
                  (default: <repo>/datasets/MMLU/data/data)

Example
-------
  python answer_position_bias.py --split test --per_subject
  python answer_position_bias.py --data_dir /content/data --split test
"""

import os
import sys
import glob
import argparse
import math
import pandas as pd

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utilities import add_data_dir_arg, resolve_data_dir  # noqa: E402
from subject_bias_analysis import CATEGORIES, get_category  # reuse existing category map

LABELS = ['A', 'B', 'C', 'D']


def load_all_splits(data_dir: str, split: str) -> pd.DataFrame:
    split_dir = os.path.join(data_dir, split)
    csv_files = glob.glob(os.path.join(split_dir, "*.csv"))
    rows = []
    for f in csv_files:
        subject = os.path.basename(f).replace(f"_{split}.csv", "")
        try:
            df = pd.read_csv(f, header=None, names=['question', 'A', 'B', 'C', 'D', 'label'])
            df['subject'] = subject
            df['category'] = get_category(subject)
            df['split'] = split
            rows.append(df)
        except Exception as e:
            print(f"Skipping {f}: {e}")
    if not rows:
        return pd.DataFrame()
    combined = pd.concat(rows, ignore_index=True)
    combined['label'] = combined['label'].astype(str).str.strip()
    return combined[combined['label'].isin(LABELS)]


def chi_square_goodness_of_fit(counts: dict, total: int):
    """Chi-square test against a uniform expected distribution (25% each)."""
    expected = total / 4.0
    if expected == 0:
        return None, None
    chi2 = sum(((counts.get(l, 0) - expected) ** 2) / expected for l in LABELS)
    # chi-square with 3 degrees of freedom -> approximate p-value via
    # Wilson-Hilferty transformation (no scipy dependency)
    df = 3
    p_value = wilson_hilferty_chi2_pvalue(chi2, df)
    return chi2, p_value


def wilson_hilferty_chi2_pvalue(chi2: float, df: int) -> float:
    """Approximate upper-tail p-value for a chi-square statistic without scipy."""
    if chi2 <= 0:
        return 1.0
    h = 2.0 / (9.0 * df)
    z = ((chi2 / df) ** (1.0 / 3.0) - (1 - h)) / math.sqrt(h)
    # standard normal upper-tail via erfc
    return 0.5 * math.erfc(z / math.sqrt(2))


def build_row(df: pd.DataFrame, label: str) -> dict:
    """Compute one table row (letter %s, chi-square, p-value, significance) for a group."""
    total = len(df)
    counts = df['label'].value_counts().to_dict()
    chi2, p_value = chi_square_goodness_of_fit(counts, total)

    pcts = {l: (counts.get(l, 0) / total * 100 if total else 0.0) for l in LABELS}
    significant = 'Significant' if (p_value is not None and p_value < 0.05) else 'Not Significant'

    return {
        'Subject': label,
        'n': total,
        'A': round(pcts['A'], 2),
        'B': round(pcts['B'], 2),
        'C': round(pcts['C'], 2),
        'D': round(pcts['D'], 2),
        'Chi-square': round(chi2, 3) if chi2 is not None else None,
        'p-value': round(p_value, 4) if p_value is not None else None,
        'Significant': significant,
    }


def run(args):
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = resolve_data_dir(args.data_dir, from_file=__file__)
    print(f"Using data_dir: {data_dir}")

    df = load_all_splits(data_dir, args.split)
    if df.empty:
        print(f"No data found for split '{args.split}'. Check --split and your data_dir path.")
        return

    rows = [build_row(df, f"OVERALL ({args.split})")]
    for subject in sorted(df['subject'].unique()):
        rows.append(build_row(df[df['subject'] == subject], subject))

    table = pd.DataFrame(rows, columns=['Subject', 'n', 'A', 'B', 'C', 'D',
                                         'Chi-square', 'p-value', 'Significant'])

    if args.per_subject:
        # Sort subjects (excluding OVERALL) by skew, keep OVERALL pinned on top
        overall_row = table[table['Subject'].str.startswith('OVERALL')]
        subject_rows = table[~table['Subject'].str.startswith('OVERALL')]
        subject_rows = subject_rows.reindex(
            subject_rows[['A', 'B', 'C', 'D']].max(axis=1).sort_values(ascending=False).index
        )
        table = pd.concat([overall_row, subject_rows], ignore_index=True)

    print("\n" + table.to_string(index=False))

    out_dir = os.path.join(_script_dir, '..', 'results')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"answer_position_bias_{args.split}.csv")
    table.to_csv(out_path, index=False)
    print(f"\nSaved table to: {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Check MMLU's answer key for letter-position bias (no API calls)")
    add_data_dir_arg(parser)
    parser.add_argument('--split', type=str, default='test', choices=['test', 'val', 'dev'])
    parser.add_argument('--per_subject', action='store_true', help="Also print a per-subject skew ranking")
    run(parser.parse_args())