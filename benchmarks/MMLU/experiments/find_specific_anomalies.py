"""
find_specific_anomalies.py

Hunt for concrete MMLU data issues and dump them to CSV:
  - missing / blank fields
  - identical option text within a question
  - cross-split leakage (same question text in test and val)

Arguments
---------
  None (CLI). Data path is inferred as:
    <repo>/datasets/MMLU/data/data

Outputs (under benchmarks/MMLU/results/)
----------------------------------------
  mmlu_missing_values.csv
  mmlu_identical_options.csv
  mmlu_cross_split_leakage_test_val.csv

Example
-------
  python find_specific_anomalies.py
"""

import os
import glob
import pandas as pd


def find_specific_anomalies(base_dir):
    splits = ['dev', 'val', 'test']
    split_dirs = {s: os.path.join(base_dir, s) for s in splits}
    results = {}

    # Collect rows from each split
    for split, path in split_dirs.items():
        if not os.path.exists(path):
            continue

        csv_files = glob.glob(os.path.join(path, "*.csv"))
        all_rows = []

        for f in csv_files:
            subject = os.path.basename(f).replace(f"_{split}.csv", "")
            try:
                df = pd.read_csv(f, header=None, names=['question', 'A', 'B', 'C', 'D', 'label'])
                df['subject'] = subject
                df['file'] = os.path.basename(f)
                df['filepath'] = f
                df['split'] = split
                df['line_no'] = range(1, len(df) + 1)  # 1-based line number
                all_rows.append(df)
            except Exception as e:
                print(f"Error reading {f}: {e}")

        if all_rows:
            results[split] = pd.concat(all_rows, ignore_index=True)

    # Prepare output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.abspath(os.path.join(script_dir, '..', 'results'))
    os.makedirs(results_dir, exist_ok=True)

    missing_rows = []
    identical_option_rows = []
    leakage_rows = []

    print("### MISSING VALUES DETECTED")
    for split, df in results.items():
        missing = df[df[['question', 'A', 'B', 'C', 'D', 'label']].isnull().any(axis=1)]
        if len(missing) > 0:
            print(f"\nSplit: {split}")
            for idx, row in missing.iterrows():
                print(f"File: {row['file']} Line: {row['line_no']}")
                print(f"Row content: {row.to_dict()}")

                missing_rows.append({
                    "split": split,
                    "file": row["file"],
                    "line_no": row["line_no"],
                    "filepath": row["filepath"],
                    "question": row["question"],
                    "A": row["A"],
                    "B": row["B"],
                    "C": row["C"],
                    "D": row["D"],
                    "label": row["label"],
                    "subject": row["subject"]
                })

    print("\n### IDENTICAL OPTIONS DETECTED")
    for split, df in results.items():
        def get_dup_choices(row):
            choices = [str(row[c]).strip() for c in ['A', 'B', 'C', 'D']]
            seen = set()
            dups = [x for x in choices if x in seen or seen.add(x)]
            return dups if len(dups) > 0 else None

        df = df.copy()
        df['dups'] = df.apply(get_dup_choices, axis=1)
        dup_rows = df[df['dups'].notnull()]

        if len(dup_rows) > 0:
            print(f"\nSplit: {split}")
            for idx, row in dup_rows.head(5).iterrows():
                print(f"File: {row['file']} Line: {row['line_no']}")
                print(f"Question: {row['question'][:100]}")
                print(f"A: {row['A']} | B: {row['B']} | C: {row['C']} | D: {row['D']} | Label: {row['label']}")

            for idx, row in dup_rows.iterrows():
                identical_option_rows.append({
                    "split": split,
                    "file": row["file"],
                    "line_no": row["line_no"],
                    "filepath": row["filepath"],
                    "question": row["question"],
                    "A": row["A"],
                    "B": row["B"],
                    "C": row["C"],
                    "D": row["D"],
                    "label": row["label"],
                    "subject": row["subject"],
                    "duplicate_options": ", ".join(row["dups"]) if isinstance(row["dups"], list) else str(row["dups"])
                })

    print("\n### CROSS-SPLIT LEAKAGE SAMPLES")
    if 'test' in results and 'val' in results:
        test_df = results['test'].copy()
        val_df = results['val'].copy()

        test_df['q_clean'] = test_df['question'].astype(str).str.strip().str.lower()
        val_df['q_clean'] = val_df['question'].astype(str).str.strip().str.lower()

        leakage = pd.merge(test_df, val_df, on='q_clean', suffixes=('_test', '_val'))
        print(f"Total overlapping questions test-val: {len(leakage)}")

        for idx, row in leakage.head(5).iterrows():
            print(f"\nLeakage item {idx + 1}:")
            print(f"Question: {row['question_test'][:120]}")
            print(f"Test File: {row['file_test']} Line: {row['line_no_test']} | Label: {row['label_test']}")
            print(f"Val File: {row['file_val']} Line: {row['line_no_val']} | Label: {row['label_val']}")

        for idx, row in leakage.iterrows():
            leakage_rows.append({
                "q_clean": row["q_clean"],
                "question_test": row["question_test"],
                "question_val": row["question_val"],
                "test_file": row["file_test"],
                "test_line_no": row["line_no_test"],
                "test_filepath": row["filepath_test"],
                "test_label": row["label_test"],
                "val_file": row["file_val"],
                "val_line_no": row["line_no_val"],
                "val_filepath": row["filepath_val"],
                "val_label": row["label_val"],
            })

    # Save CSV outputs
    if missing_rows:
        pd.DataFrame(missing_rows).to_csv(
            os.path.join(results_dir, "mmlu_missing_values.csv"),
            index=False
        )

    if identical_option_rows:
        pd.DataFrame(identical_option_rows).to_csv(
            os.path.join(results_dir, "mmlu_identical_options.csv"),
            index=False
        )

    if leakage_rows:
        pd.DataFrame(leakage_rows).to_csv(
            os.path.join(results_dir, "mmlu_cross_split_leakage_test_val.csv"),
            index=False
        )

    print(f"\nSaved CSV files to: {results_dir}")


if __name__ == '__main__':
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.abspath(os.path.join(_script_dir, '..', '..', '..'))
    base_dir = os.path.join(_project_root, 'datasets', 'MMLU', 'data', 'data')
    find_specific_anomalies(base_dir)