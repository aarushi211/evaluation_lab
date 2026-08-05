"""
negation_impact.py

Research Question 4: Does negation hurt LLM performance?

Methodology
-----------
Negation words ("not", "except", "false", "incorrect", "wrong") are NOT evenly
distributed across MMLU subjects (Humanities ~44% vs STEM ~7%, per
analysis/dataset_limitations.md). A naive global comparison of
negation-vs-non-negation accuracy is confounded by subject difficulty.

To isolate the negation effect:
1. For each subject, split test questions into has_negation / no_negation groups.
2. Only use subjects that have at least `--min_per_group` questions in BOTH
   groups (so we're comparing negation vs non-negation within the same topic).
3. Sample up to `--limit_per_group` questions from each group per subject.
4. Evaluate all sampled questions with the same model/settings.
5. Report accuracy split overall, per-subject, and a two-proportion z-test
   for the overall difference.

Checkpointing
-------------
Every question's result is appended to the output CSV immediately after it's
evaluated (not buffered until the end). If the script is interrupted (e.g. a
Groq free-tier rate limit kills the process), just rerun the exact same
command: it detects already-completed question_ids in the output file and
skips them, picking up where it left off.

This reuses LLMEvaluator / format_question / extract_answer from evaluate_mmlu.py
so results stay comparable with your other experiments (same prompting logic).
"""

import os
import sys
import glob
import random
import argparse
import math
import hashlib
import csv
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_mmlu import LLMEvaluator, format_question, extract_answer  # noqa: E402

NEGATION_PATTERN = r'\b(?:not|except|false|incorrect|wrong)\b'

FIELDNAMES = [
    'question_id', 'subject', 'has_negation', 'question',
    'ground_truth', 'predicted', 'raw_output', 'correct',
]


def make_question_id(subject: str, question: str) -> str:
    """Stable content-based ID so resuming works even if sampling order changes."""
    return hashlib.md5(f"{subject}::{question}".encode('utf-8')).hexdigest()[:12]


def load_subject_df(data_dir: str, subject: str) -> pd.DataFrame:
    test_file = os.path.join(data_dir, 'test', f"{subject}_test.csv")
    if not os.path.exists(test_file):
        return pd.DataFrame()
    df = pd.read_csv(test_file, header=None, names=['question', 'A', 'B', 'C', 'D', 'label'])
    df['subject'] = subject
    df['label'] = df['label'].astype(str).str.strip()
    df['has_negation'] = df['question'].astype(str).str.lower().str.contains(NEGATION_PATTERN, regex=True)
    return df


def build_sample(data_dir: str, subjects, limit_per_group: int, min_per_group: int, seed: int) -> pd.DataFrame:
    all_samples = []

    for subject in subjects:
        df = load_subject_df(data_dir, subject)
        if df.empty:
            continue

        neg_df = df[df['has_negation']]
        non_neg_df = df[~df['has_negation']]

        if len(neg_df) < min_per_group or len(non_neg_df) < min_per_group:
            continue  # subject doesn't have enough of both groups to compare fairly

        neg_sample = neg_df.sample(n=min(limit_per_group, len(neg_df)), random_state=seed)
        non_neg_sample = non_neg_df.sample(n=min(limit_per_group, len(non_neg_df)), random_state=seed)

        all_samples.append(neg_sample)
        all_samples.append(non_neg_sample)

    if not all_samples:
        return pd.DataFrame()

    combined = pd.concat(all_samples, ignore_index=True)
    combined = combined.sample(frac=1.0, random_state=seed).reset_index(drop=True)  # shuffle call order
    combined['question_id'] = combined.apply(lambda r: make_question_id(r['subject'], r['question']), axis=1)
    return combined


def load_processed_ids(out_path: str) -> set:
    if not os.path.exists(out_path):
        return set()
    try:
        existing = pd.read_csv(out_path)
        return set(existing['question_id'].astype(str))
    except Exception:
        return set()


def append_result_row(out_path: str, row: dict):
    file_exists = os.path.exists(out_path)
    with open(out_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def two_proportion_z_test(correct_a, total_a, correct_b, total_b):
    """Two-sided z-test for difference in accuracy between two independent groups."""
    if total_a == 0 or total_b == 0:
        return None, None
    p_a = correct_a / total_a
    p_b = correct_b / total_b
    p_pool = (correct_a + correct_b) / (total_a + total_b)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / total_a + 1 / total_b))
    if se == 0:
        return 0.0, 1.0
    z = (p_a - p_b) / se
    p_value = math.erfc(abs(z) / math.sqrt(2))
    return z, p_value


def run(args):
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.abspath(os.path.join(_script_dir, '..', '..', '..'))
    data_dir = os.path.join(_project_root, 'datasets', 'MMLU', 'data', 'data')

    if args.subjects.strip().lower() == 'all':
        test_dir = os.path.join(data_dir, 'test')
        subjects = sorted(
            os.path.basename(f).replace('_test.csv', '')
            for f in glob.glob(os.path.join(test_dir, '*.csv'))
        )
    else:
        subjects = [s.strip() for s in args.subjects.split(',') if s.strip()]

    sample_df = build_sample(data_dir, subjects, args.limit_per_group, args.min_per_group, args.seed)
    if sample_df.empty:
        print("No eligible subjects found (need enough questions in both negation groups). Try lowering --min_per_group.")
        return

    out_dir = os.path.join(_script_dir, '..', 'results')
    os.makedirs(out_dir, exist_ok=True)
    out_name = f"negation_{args.provider}_{args.model.replace('/', '_').replace('.', '_')}_lim{args.limit_per_group}.csv"
    out_path = os.path.join(out_dir, out_name)

    processed_ids = load_processed_ids(out_path)
    remaining = sample_df[~sample_df['question_id'].isin(processed_ids)]

    print(f"Total sample: {len(sample_df)} questions across {sample_df['subject'].nunique()} subjects "
          f"({sample_df['has_negation'].sum()} negation / {(~sample_df['has_negation']).sum()} non-negation).")
    if processed_ids:
        print(f"Found existing results at {out_path} — {len(processed_ids)} already done, "
              f"{len(remaining)} remaining. Resuming.")

    if len(remaining) == 0:
        print("Nothing left to evaluate — all questions already have results.")
    else:
        evaluator = LLMEvaluator(provider=args.provider, model_name=args.model, api_key=args.api_key, use_json=args.json)

        for _, row in remaining.iterrows():
            options = [str(row['A']), str(row['B']), str(row['C']), str(row['D'])]
            prompt = format_question(row['question'], options)
            raw_output = evaluator.query(prompt)
            pred_label = extract_answer(raw_output)
            is_correct = (pred_label == row['label'])

            result_row = {
                'question_id': row['question_id'],
                'subject': row['subject'],
                'has_negation': row['has_negation'],
                'question': row['question'],
                'ground_truth': row['label'],
                'predicted': pred_label,
                'raw_output': raw_output,
                'correct': is_correct,
            }
            append_result_row(out_path, result_row)

            print(f"[{row['subject']}] negation={row['has_negation']} | GT={row['label']} | "
                  f"Pred={pred_label} | {'Correct' if is_correct else 'Incorrect'}")

    # Reload full results (old + new) from disk for final stats
    results_df = pd.read_csv(out_path)
    results_df['has_negation'] = results_df['has_negation'].astype(bool)
    results_df['correct'] = results_df['correct'].astype(bool)

    neg_results = results_df[results_df['has_negation']]
    non_neg_results = results_df[~results_df['has_negation']]

    neg_correct, neg_total = neg_results['correct'].sum(), len(neg_results)
    non_neg_correct, non_neg_total = non_neg_results['correct'].sum(), len(non_neg_results)

    neg_acc = (neg_correct / neg_total * 100) if neg_total else 0
    non_neg_acc = (non_neg_correct / non_neg_total * 100) if non_neg_total else 0

    z, p_value = two_proportion_z_test(neg_correct, neg_total, non_neg_correct, non_neg_total)

    print("\n=== OVERALL RESULTS ===")
    print(f"Negation questions:     {neg_acc:.2f}% ({neg_correct}/{neg_total})")
    print(f"Non-negation questions: {non_neg_acc:.2f}% ({non_neg_correct}/{non_neg_total})")
    if z is not None:
        print(f"Two-proportion z-test: z={z:.3f}, p={p_value:.4f} "
              f"({'significant at p<0.05' if p_value < 0.05 else 'not significant at p<0.05'})")

    print("\n=== PER-SUBJECT BREAKDOWN ===")
    subject_summary = results_df.groupby(['subject', 'has_negation'])['correct'].agg(['mean', 'count']).reset_index()
    subject_summary['mean'] = (subject_summary['mean'] * 100).round(2)
    print(subject_summary.to_string(index=False))

    print(f"\nResults file (append-only, safe to resume): {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test whether negation in questions hurts LLM accuracy on MMLU")
    parser.add_argument('--provider', type=str, default='ollama', choices=['ollama', 'groq'])
    parser.add_argument('--model', type=str, default='llama3.2')
    parser.add_argument('--subjects', type=str, default='all',
                         help="Comma-separated subject list, or 'all' to scan every subject")
    parser.add_argument('--limit_per_group', type=int, default=20,
                         help="Max questions sampled per negation/non-negation group per subject")
    parser.add_argument('--min_per_group', type=int, default=10,
                         help="Minimum questions required in BOTH groups for a subject to be included")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--api_key', type=str, default=None)
    parser.add_argument('--json', action='store_true')

    run(parser.parse_args())