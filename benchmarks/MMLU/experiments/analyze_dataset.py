import os
import glob
import pandas as pd
import numpy as np


def analyze_mmlu(base_dir):
    splits = ['dev', 'val', 'test']
    split_dirs = {s: os.path.join(base_dir, s) for s in splits}

    results = {}
    summary_rows = []

    print("--- Starting MMLU Dataset Analysis ---")

    for split, path in split_dirs.items():
        if not os.path.exists(path):
            print(f"Path does not exist: {path}")
            continue

        csv_files = glob.glob(os.path.join(path, "*.csv"))
        print(f"Analyzing split '{split}': found {len(csv_files)} files.")

        all_rows = []
        for f in csv_files:
            subject = os.path.basename(f).replace(f"_{split}.csv", "")
            try:
                # No header in MMLU files
                df = pd.read_csv(f, header=None, names=['question', 'A', 'B', 'C', 'D', 'label'])
                df['subject'] = subject
                df['file'] = os.path.basename(f)
                all_rows.append(df)
            except Exception as e:
                print(f"Error reading {f}: {e}")

        if not all_rows:
            continue

        df_split = pd.concat(all_rows, ignore_index=True)
        results[split] = df_split

    print("\n--- Basic Statistics ---")
    for split, df in results.items():
        total_questions = len(df)
        distinct_subjects = df['subject'].nunique()

        print(f"Split: {split}")
        print(f"  Total questions: {total_questions}")
        print(f"  Distinct subjects: {distinct_subjects}")

        summary_rows.append({
            "section": "basic_statistics",
            "split": split,
            "metric": "total_questions",
            "value": total_questions,
            "notes": ""
        })
        summary_rows.append({
            "section": "basic_statistics",
            "split": split,
            "metric": "distinct_subjects",
            "value": distinct_subjects,
            "notes": ""
        })

    print("\n--- Label Distribution ---")
    for split, df in results.items():
        print(f"Split: {split}")
        dist = df['label'].astype(str).str.strip().value_counts(normalize=True) * 100
        for label in ['A', 'B', 'C', 'D']:
            val = float(dist.get(label, 0.0))
            print(f"  {label}: {val:.2f}%")

            summary_rows.append({
                "section": "label_distribution",
                "split": split,
                "metric": f"label_{label}_pct",
                "value": val,
                "notes": ""
            })

    print("\n--- Option Length Bias Analysis ---")
    for split, df in results.items():
        print(f"Split: {split}")

        df = df.copy()
        df['label'] = df['label'].astype(str).str.strip()

        for col in ['A', 'B', 'C', 'D']:
            df[f'{col}_len'] = df[col].astype(str).str.len()

        def is_correct_longest(row):
            lbl = row['label']
            if lbl not in ['A', 'B', 'C', 'D']:
                return np.nan
            lengths = {c: row[f'{c}_len'] for c in ['A', 'B', 'C', 'D']}
            max_len = max(lengths.values())
            return 1 if lengths[lbl] == max_len else 0

        def correct_vs_avg_incorrect(row):
            lbl = row['label']
            if lbl not in ['A', 'B', 'C', 'D']:
                return np.nan
            correct_len = row[f'{lbl}_len']
            incorrect_lens = [row[f'{c}_len'] for c in ['A', 'B', 'C', 'D'] if c != lbl]
            avg_incorrect = np.mean(incorrect_lens)
            return correct_len - avg_incorrect

        longest_flags = df.apply(is_correct_longest, axis=1)
        len_diffs = df.apply(correct_vs_avg_incorrect, axis=1)

        longest_pct = float(longest_flags.mean() * 100)
        avg_diff = float(len_diffs.mean())

        print(f"  Percentage of times correct option is the longest: {longest_pct:.2f}% (random chance is ~25%)")
        print(f"  Average length difference (correct - average incorrect): {avg_diff:.2f} characters")

        summary_rows.append({
            "section": "option_length_bias",
            "split": split,
            "metric": "correct_option_longest_pct",
            "value": longest_pct,
            "notes": "Random chance ~25%"
        })
        summary_rows.append({
            "section": "option_length_bias",
            "split": split,
            "metric": "avg_length_diff_correct_minus_avg_incorrect",
            "value": avg_diff,
            "notes": ""
        })

    print("\n--- Data Overlap / Duplicate Analysis ---")
    for split, df in results.items():
        dups = df[df.duplicated(subset=['question'], keep=False)]
        dup_count = len(dups)

        print(f"Split '{split}': found {dup_count} duplicate/near-duplicate questions based on exact question text.")
        if dup_count > 0:
            print("  Example duplicate question:")
            print(f"    {dups.iloc[0]['question'][:150]}...")

        summary_rows.append({
            "section": "duplicate_analysis",
            "split": split,
            "metric": "duplicate_question_count",
            "value": dup_count,
            "notes": dups.iloc[0]['question'][:150] + "..." if dup_count > 0 else ""
        })

    if 'test' in results and 'val' in results and 'dev' in results:
        test_qs = set(results['test']['question'].astype(str).str.strip())
        val_qs = set(results['val']['question'].astype(str).str.strip())
        dev_qs = set(results['dev']['question'].astype(str).str.strip())

        overlap_test_val = test_qs.intersection(val_qs)
        overlap_test_dev = test_qs.intersection(dev_qs)
        overlap_val_dev = val_qs.intersection(dev_qs)

        print(f"Leakage check:")
        print(f"  Overlap between 'test' and 'val': {len(overlap_test_val)} questions")
        print(f"  Overlap between 'test' and 'dev': {len(overlap_test_dev)} questions")
        print(f"  Overlap between 'val' and 'dev': {len(overlap_val_dev)} questions")

        summary_rows.extend([
            {
                "section": "cross_split_overlap",
                "split": "test_val",
                "metric": "overlap_count",
                "value": len(overlap_test_val),
                "notes": ""
            },
            {
                "section": "cross_split_overlap",
                "split": "test_dev",
                "metric": "overlap_count",
                "value": len(overlap_test_dev),
                "notes": ""
            },
            {
                "section": "cross_split_overlap",
                "split": "val_dev",
                "metric": "overlap_count",
                "value": len(overlap_val_dev),
                "notes": ""
            },
        ])

    print("\n--- Formatting & Value Anomalies ---")
    for split, df in results.items():
        print(f"Split: {split}")

        df = df.copy()
        df['label'] = df['label'].astype(str).str.strip()

        invalid_labels = df[~df['label'].isin(['A', 'B', 'C', 'D'])]
        print(f"  Questions with invalid labels: {len(invalid_labels)}")

        missing_vals = df[df[['question', 'A', 'B', 'C', 'D', 'label']].isnull().any(axis=1)]
        print(f"  Questions with missing values: {len(missing_vals)}")

        def duplicate_choices(row):
            choices = [str(row[c]).strip() for c in ['A', 'B', 'C', 'D']]
            return len(choices) != len(set(choices))

        dup_choices_flags = df.apply(duplicate_choices, axis=1)
        dup_choice_count = int(dup_choices_flags.sum())
        print(f"  Questions with identical options (e.g. A and B are the same): {dup_choice_count}")

        summary_rows.append({
            "section": "formatting_anomalies",
            "split": split,
            "metric": "invalid_label_count",
            "value": len(invalid_labels),
            "notes": ""
        })
        summary_rows.append({
            "section": "formatting_anomalies",
            "split": split,
            "metric": "missing_value_count",
            "value": len(missing_vals),
            "notes": ""
        })
        summary_rows.append({
            "section": "formatting_anomalies",
            "split": split,
            "metric": "identical_option_count",
            "value": dup_choice_count,
            "notes": ""
        })

    # Save CSV summary
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.abspath(os.path.join(script_dir, '..', 'results'))
    os.makedirs(results_dir, exist_ok=True)

    output_path = os.path.join(results_dir, 'mmlu_analysis_results.csv')
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_path, index=False)

    print(f"\nSaved CSV summary to: {output_path}")


if __name__ == '__main__':
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.abspath(os.path.join(_script_dir, '..', '..', '..'))
    base_dir = os.path.join(_project_root, 'datasets', 'MMLU', 'data', 'data')
    analyze_mmlu(base_dir)