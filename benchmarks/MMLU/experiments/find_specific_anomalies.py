import os
import glob
import pandas as pd

def find_specific_anomalies(base_dir):
    splits = ['dev', 'val', 'test']
    split_dirs = {s: os.path.join(base_dir, s) for s in splits}
    results = {}
    
    for split, path in split_dirs.items():
        csv_files = glob.glob(os.path.join(path, "*.csv"))
        all_rows = []
        for f in csv_files:
            subject = os.path.basename(f).replace(f"_{split}.csv", "")
            try:
                df = pd.read_csv(f, header=None, names=['question', 'A', 'B', 'C', 'D', 'label'])
                df['subject'] = subject
                df['file'] = os.path.basename(f)
                df['filepath'] = f
                # Add line number (1-based because CSV headerless starts at 1)
                df['line_no'] = range(1, len(df) + 1)
                all_rows.append(df)
            except Exception as e:
                pass
        if all_rows:
            results[split] = pd.concat(all_rows, ignore_index=True)
            
    print("### MISSING VALUES DETECTED")
    for split, df in results.items():
        missing = df[df[['question', 'A', 'B', 'C', 'D', 'label']].isnull().any(axis=1)]
        if len(missing) > 0:
            print(f"\nSplit: {split}")
            for idx, row in missing.iterrows():
                print(f"File: {row['file']} Line: {row['line_no']}")
                print(f"Row content: {row.to_dict()}")

    print("\n### IDENTICAL OPTIONS DETECTED")
    for split, df in results.items():
        # helper to find duplicate options
        def get_dup_choices(row):
            choices = [str(row[c]).strip() for c in ['A', 'B', 'C', 'D']]
            seen = set()
            dups = [x for x in choices if x in seen or seen.add(x)]
            return dups if len(dups) > 0 else None
            
        df['dups'] = df.apply(get_dup_choices, axis=1)
        dup_rows = df[df['dups'].notnull()]
        if len(dup_rows) > 0:
            print(f"\nSplit: {split}")
            for idx, row in dup_rows.head(5).iterrows():
                print(f"File: {row['file']} Line: {row['line_no']}")
                print(f"Question: {row['question'][:100]}")
                print(f"A: {row['A']} | B: {row['B']} | C: {row['C']} | D: {row['D']} | Label: {row['label']}")

    print("\n### CROSS-SPLIT LEAKAGE SAMPLES")
    if 'test' in results and 'val' in results:
        test_df = results['test']
        val_df = results['val']
        # overlap by question text (normalized)
        test_df['q_clean'] = test_df['question'].astype(str).str.strip().str.lower()
        val_df['q_clean'] = val_df['question'].astype(str).str.strip().str.lower()
        
        leakage = pd.merge(test_df, val_df, on='q_clean', suffixes=('_test', '_val'))
        print(f"Total overlapping questions test-val: {len(leakage)}")
        for idx, row in leakage.head(5).iterrows():
            print(f"\nLeakage item {idx+1}:")
            print(f"Question: {row['question_test'][:120]}")
            print(f"Test File: {row['file_test']} Line: {row['line_no_test']} | Label: {row['label_test']}")
            print(f"Val File: {row['file_val']} Line: {row['line_no_val']} | Label: {row['label_val']}")

if __name__ == '__main__':
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.abspath(os.path.join(_script_dir, '..', '..', '..'))
    base_dir = os.path.join(_project_root, 'datasets', 'MMLU', 'data', 'data')
    find_specific_anomalies(base_dir)
