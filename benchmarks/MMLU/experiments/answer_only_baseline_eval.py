"""
answer_only_baseline_eval.py

Check: Can a model beat random chance (25%) using ONLY the four options,
with the question text removed entirely?

If it can, that means the model is exploiting structural/phrasing cues in
the options themselves (length, specificity, "sounds like a distractor"
patterns) rather than reasoning about the actual question — a stronger,
more damning claim than length bias alone, since here there is nothing
correct to reason about at all.

This reuses LLMEvaluator / extract_answer from evaluate_mmlu.py, but uses
its own prompt (no question shown) instead of format_question().

Checkpointing
-------------
Same pattern as negation_impact_eval.py / typo_robustness_eval.py: each
result is appended to the output CSV immediately, and rerunning the exact
same command auto-resumes by skipping already-completed question_ids.
"""

import os
import sys
import glob
import argparse
import math
import hashlib
import csv
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_mmlu import LLMEvaluator, extract_answer  # noqa: E402

FIELDNAMES = ['question_id', 'subject', 'options_only_prompt', 'ground_truth', 'predicted', 'raw_output', 'correct']


def make_question_id(subject: str, question: str) -> str:
    return hashlib.md5(f"{subject}::{question}".encode('utf-8')).hexdigest()[:12]


def format_options_only(options):
    prompt = "Question: [omitted]\n"
    prompt += f"A. {options[0]}\n"
    prompt += f"B. {options[1]}\n"
    prompt += f"C. {options[2]}\n"
    prompt += f"D. {options[3]}\n"
    prompt += "Without seeing the question, which option is most likely to be correct? Answer:"
    return prompt


def build_sample(data_dir: str, subjects, total_limit: int, seed: int) -> pd.DataFrame:
    all_rows = []
    for subject in subjects:
        test_file = os.path.join(data_dir, 'test', f"{subject}_test.csv")
        if not os.path.exists(test_file):
            continue
        df = pd.read_csv(test_file, header=None, names=['question', 'A', 'B', 'C', 'D', 'label'])
        df['subject'] = subject
        df['label'] = df['label'].astype(str).str.strip()
        all_rows.append(df)

    if not all_rows:
        return pd.DataFrame()

    combined = pd.concat(all_rows, ignore_index=True)
    combined = combined[combined['label'].isin(['A', 'B', 'C', 'D'])]
    sample = combined.sample(n=min(total_limit, len(combined)), random_state=seed).reset_index(drop=True)
    sample['question_id'] = sample.apply(lambda r: make_question_id(r['subject'], r['question']), axis=1)
    return sample


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


def one_sample_proportion_test(correct: int, total: int, null_p: float = 0.25):
    """Test whether observed accuracy differs from the chance baseline (null_p)."""
    if total == 0:
        return None, None
    p_hat = correct / total
    se = math.sqrt(null_p * (1 - null_p) / total)
    if se == 0:
        return 0.0, 1.0
    z = (p_hat - null_p) / se
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

    sample_df = build_sample(data_dir, subjects, args.limit, args.seed)
    if sample_df.empty:
        print("No questions found for the given subjects.")
        return

    out_dir = os.path.join(_script_dir, '..', 'results')
    os.makedirs(out_dir, exist_ok=True)
    out_name = f"answer_only_{args.provider}_{args.model.replace('/', '_').replace('.', '_')}_lim{args.limit}.csv"
    out_path = os.path.join(out_dir, out_name)

    processed_ids = load_processed_ids(out_path)
    remaining = sample_df[~sample_df['question_id'].isin(processed_ids)]

    print(f"Total sample: {len(sample_df)} questions across {sample_df['subject'].nunique()} subjects. "
          f"Question text will be hidden from the model.")
    if processed_ids:
        print(f"Found existing results at {out_path} — {len(processed_ids)} already done, "
              f"{len(remaining)} remaining. Resuming.")

    if len(remaining) == 0:
        print("Nothing left to evaluate — all questions already have results.")
    else:
        evaluator = LLMEvaluator(provider=args.provider, model_name=args.model, api_key=args.api_key, use_json=args.json)

        for _, row in remaining.iterrows():
            options = [str(row['A']), str(row['B']), str(row['C']), str(row['D'])]
            prompt = format_options_only(options)
            raw_output = evaluator.query(prompt)
            pred_label = extract_answer(raw_output)
            is_correct = (pred_label == row['label'])

            result_row = {
                'question_id': row['question_id'],
                'subject': row['subject'],
                'options_only_prompt': prompt.replace('\n', ' | '),
                'ground_truth': row['label'],
                'predicted': pred_label,
                'raw_output': raw_output,
                'correct': is_correct,
            }
            append_result_row(out_path, result_row)

            print(f"[{row['subject']}] GT={row['label']} | Pred={pred_label} | "
                  f"{'Correct' if is_correct else 'Incorrect'}")

    results_df = pd.read_csv(out_path)
    results_df['correct'] = results_df['correct'].astype(bool)

    total = len(results_df)
    correct = results_df['correct'].sum()
    acc = correct / total * 100 if total else 0
    z, p_value = one_sample_proportion_test(correct, total, null_p=0.25)

    print("\n=== OVERALL RESULTS ===")
    print(f"Options-only accuracy: {acc:.2f}% ({correct}/{total})  [random chance = 25%]")
    if z is not None:
        print(f"One-sample proportion test vs. 25% chance: z={z:.3f}, p={p_value:.4f} "
              f"({'SIGNIFICANTLY above/below chance' if p_value < 0.05 else 'not significantly different from chance'})")

    print("\n=== PER-SUBJECT BREAKDOWN ===")
    subj_summary = results_df.groupby('subject')['correct'].agg(['mean', 'count']).reset_index()
    subj_summary['mean'] = (subj_summary['mean'] * 100).round(2)
    subj_summary = subj_summary.sort_values('mean', ascending=False)
    print(subj_summary.to_string(index=False))

    print(f"\nResults file (append-only, safe to resume): {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Test whether a model can beat chance using ONLY the options, question text hidden"
    )
    parser.add_argument('--provider', type=str, default='ollama', choices=['ollama', 'groq'])
    parser.add_argument('--model', type=str, default='llama3.2')
    parser.add_argument('--subjects', type=str, default='all',
                         help="Comma-separated subject list, or 'all' to sample across every subject")
    parser.add_argument('--limit', type=int, default=100, help="Total number of questions to sample")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--api_key', type=str, default=None)
    parser.add_argument('--json', action='store_true')

    run(parser.parse_args())