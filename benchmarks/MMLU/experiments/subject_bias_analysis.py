"""
subject_bias_analysis.py

Measure option-length / position cues and negation density by MMLU subject
and high-level academic category (STEM / Humanities / Social Sciences / Other).

Writes category and subject summary CSVs under benchmarks/MMLU/results/.
Also exports CATEGORIES / get_category for reuse by other dataset scripts.

Arguments
---------
  --data_dir   Folder containing test/ — Colab-friendly
               (default: <repo>/datasets/MMLU/data/data)

Example
-------
  python subject_bias_analysis.py
  python subject_bias_analysis.py --data_dir /content/data
"""

import os
import sys
import glob
import argparse
import pandas as pd
import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utilities import add_data_dir_arg, resolve_data_dir  # noqa: E402

# MMLU High-Level Category Mapping
CATEGORIES = {
    'STEM': [
        'abstract_algebra', 'astronomy', 'college_biology', 'college_chemistry',
        'college_computer_science', 'college_mhematics', 'college_physics',
        'computer_security', 'conceptual_physics', 'electrical_engineering',
        'elementary_mathematics', 'formal_logic', 'high_school_biology',
        'high_school_chemistry', 'high_school_computer_science',
        'high_school_mathematics', 'high_school_physics', 'high_school_statistics',
        'machine_learning', 'medical_genetics', 'virology'
    ],
    'Humanities': [
        'business_ethics', 'high_school_european_history', 'high_school_us_history',
        'high_school_world_history', 'history_of_science', 'international_law',
        'jurisprudence', 'logical_fallacies', 'moral_disputes', 'moral_scenarios',
        'philosophy', 'prehistory', 'world_religions'
    ],
    'Social Sciences': [
        'econometrics', 'high_school_geography', 'high_school_government_and_politics',
        'high_school_macroeconomics', 'high_school_microeconomics', 'high_school_psychology',
        'human_sexuality', 'professional_psychology', 'sociology', 'us_foreign_policy'
    ],
    'Other (Applied/Professional)': [
        'anatomy', 'clinical_knowledge', 'college_medicine', 'global_facts',
        'human_aging', 'management', 'marketing', 'medical_genetics',
        'miscellaneous', 'nutrition', 'professional_accounting', 'professional_law',
        'professional_medicine', 'public_relations', 'security_studies'
    ]
}

def get_category(subject):
    for cat, subs in CATEGORIES.items():
        if subject in subs:
            return cat
    return 'Other (Applied/Professional)'

def analyze_subject_biases(base_dir):
    test_path = os.path.join(base_dir, 'test')
    csv_files = glob.glob(os.path.join(test_path, "*.csv"))

    all_rows = []
    for f in csv_files:
        subject = os.path.basename(f).replace("_test.csv", "")
        try:
            df = pd.read_csv(f, header=None, names=['question', 'A', 'B', 'C', 'D', 'label'])
            df['subject'] = subject
            df['category'] = get_category(subject)
            all_rows.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")

    if not all_rows:
        print("No CSV files were loaded successfully.")
        return

    df_all = pd.concat(all_rows, ignore_index=True)
    df_all['label'] = df_all['label'].astype(str).str.strip()

    # Calculate option lengths
    for col in ['A', 'B', 'C', 'D']:
        df_all[f'{col}_len'] = df_all[col].astype(str).str.len()

    def is_correct_longest(row):
        lbl = row['label']
        if lbl not in ['A', 'B', 'C', 'D']:
            return np.nan
        lengths = {c: row[f'{c}_len'] for c in ['A', 'B', 'C', 'D']}
        max_len = max(lengths.values())
        return 1 if lengths[lbl] == max_len else 0

    df_all['correct_longest'] = df_all.apply(is_correct_longest, axis=1)

    # Negation analysis
    negation_words = [' not ', ' except ', ' incorrect', ' false', ' wrong', ' truth value']

    def contains_negation(q):
        q_str = str(q).lower()
        return any(word in q_str for word in negation_words)

    df_all['has_negation'] = df_all['question'].apply(contains_negation)

    # Summaries
    cat_summary = df_all.groupby('category').agg(
        total_questions=('question', 'count'),
        longest_option_bias=('correct_longest', lambda x: x.mean() * 100),
        negation_questions=('has_negation', 'sum'),
        negation_percentage=('has_negation', lambda x: x.mean() * 100)
    ).reset_index()

    sub_summary = df_all.groupby(['subject', 'category']).agg(
        total_questions=('question', 'count'),
        longest_option_bias=('correct_longest', lambda x: x.mean() * 100),
        negation_questions=('has_negation', 'sum'),
        negation_percentage=('has_negation', lambda x: x.mean() * 100)
    ).reset_index()

    top_bias = sub_summary.sort_values(by='longest_option_bias', ascending=False).head(10)
    low_bias = sub_summary.sort_values(by='longest_option_bias', ascending=True).head(10)
    neg_summary = df_all.groupby('category').agg(
        total_questions=('question', 'count'),
        negation_questions=('has_negation', 'sum'),
        negation_percentage=('has_negation', lambda x: x.mean() * 100)
    ).reset_index()

    # Print summaries
    print("=== ANALYSIS BY HIGH-LEVEL CATEGORY ===")
    print(cat_summary.to_string(index=False))

    print("\n=== TOP 10 SUBJECTS WITH HIGHEST OPTION LENGTH BIAS ===")
    print(top_bias.to_string(index=False))

    print("\n=== TOP 10 SUBJECTS WITH LOWEST OPTION LENGTH BIAS ===")
    print(low_bias.to_string(index=False))

    print("\n=== NEGATIVE/EXCEPT QUESTIONS BY CATEGORY ===")
    print(neg_summary.to_string(index=False))

    # Save results
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.abspath(os.path.join(script_dir, '..', 'results'))
    os.makedirs(results_dir, exist_ok=True)

    cat_summary.to_csv(os.path.join(results_dir, 'mmlu_category_summary.csv'), index=False)
    sub_summary.to_csv(os.path.join(results_dir, 'mmlu_subject_bias_summary.csv'), index=False)
    top_bias.to_csv(os.path.join(results_dir, 'mmlu_top_10_subjects_by_bias.csv'), index=False)
    low_bias.to_csv(os.path.join(results_dir, 'mmlu_bottom_10_subjects_by_bias.csv'), index=False)
    neg_summary.to_csv(os.path.join(results_dir, 'mmlu_negation_summary.csv'), index=False)

    print(f"\nSaved CSV files to: {results_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Measure MMLU subject/category biases")
    add_data_dir_arg(parser)
    args = parser.parse_args()
    base_dir = resolve_data_dir(args.data_dir, from_file=__file__)
    print(f"Using data_dir: {base_dir}")
    analyze_subject_biases(base_dir)