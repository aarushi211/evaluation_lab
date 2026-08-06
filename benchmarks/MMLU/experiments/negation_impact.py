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
Every question's result is appended to the output CSV immediately. Rerun the
exact same command to resume from already-completed question_ids.

Arguments
---------
  --provider         {ollama, groq, openai, anthropic, gemini}  (default: ollama)
  --model            Model name / id                            (default: llama3.2)
  --api_key          API key; comma-separated for rotation      (optional; else env)
  --json             Request JSON-shaped A/B/C/D answers        (flag)

  --subjects         Comma-separated subject list, or "all"     (default: all)
  --limit_per_group  Max questions per negation group / subject (default: 20)
  --min_per_group    Min questions required in BOTH groups      (default: 10)
  --seed             Sampling RNG seed                          (default: 42)

Example
-------
  python negation_impact.py --provider anthropic --model claude-3-5-haiku-latest \\
      --subjects all --limit_per_group 20 --min_per_group 10 --json
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

from utilities import (  # noqa: E402
    LLMEvaluator,
    add_llm_args,
    append_result_row,
    dataset_dir,
    extract_answer,
    format_question,
    load_processed_ids,
    make_question_id,
    project_root,
)

NEGATION_PATTERN = r"\b(?:not|except|false|incorrect|wrong)\b"

FIELDNAMES = [
    "question_id",
    "subject",
    "has_negation",
    "question",
    "ground_truth",
    "predicted",
    "raw_output",
    "correct",
]


def load_subject_df(data_dir: str, subject: str) -> pd.DataFrame:
    test_file = os.path.join(data_dir, "test", f"{subject}_test.csv")
    if not os.path.exists(test_file):
        return pd.DataFrame()
    df = pd.read_csv(
        test_file, header=None, names=["question", "A", "B", "C", "D", "label"]
    )
    df["subject"] = subject
    df["label"] = df["label"].astype(str).str.strip()
    df["has_negation"] = (
        df["question"].astype(str).str.lower().str.contains(NEGATION_PATTERN, regex=True)
    )
    return df


def build_sample(
    data_dir: str, subjects, limit_per_group: int, min_per_group: int, seed: int
) -> pd.DataFrame:
    all_samples = []

    for subject in subjects:
        df = load_subject_df(data_dir, subject)
        if df.empty:
            continue

        neg_df = df[df["has_negation"]]
        non_neg_df = df[~df["has_negation"]]

        if len(neg_df) < min_per_group or len(non_neg_df) < min_per_group:
            continue

        neg_sample = neg_df.sample(
            n=min(limit_per_group, len(neg_df)), random_state=seed
        )
        non_neg_sample = non_neg_df.sample(
            n=min(limit_per_group, len(non_neg_df)), random_state=seed
        )

        all_samples.append(neg_sample)
        all_samples.append(non_neg_sample)

    if not all_samples:
        return pd.DataFrame()

    combined = pd.concat(all_samples, ignore_index=True)
    combined = combined.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    combined["question_id"] = combined.apply(
        lambda r: make_question_id(r["subject"], r["question"]), axis=1
    )
    return combined


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
    data_dir = dataset_dir("MMLU", "data", "data", from_file=__file__)

    if args.subjects.strip().lower() == "all":
        test_dir = os.path.join(data_dir, "test")
        subjects = sorted(
            os.path.basename(f).replace("_test.csv", "")
            for f in glob.glob(os.path.join(test_dir, "*.csv"))
        )
    else:
        subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]

    sample_df = build_sample(
        data_dir, subjects, args.limit_per_group, args.min_per_group, args.seed
    )
    if sample_df.empty:
        print(
            "No eligible subjects found (need enough questions in both negation groups). "
            "Try lowering --min_per_group."
        )
        return

    out_dir = os.path.join(project_root(__file__), "benchmarks", "MMLU", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_name = (
        f"negation_{args.provider}_{args.model.replace('/', '_').replace('.', '_')}"
        f"_lim{args.limit_per_group}.csv"
    )
    out_path = os.path.join(out_dir, out_name)

    processed_ids = load_processed_ids(out_path)
    remaining = sample_df[~sample_df["question_id"].isin(processed_ids)]

    print(
        f"Total sample: {len(sample_df)} questions across "
        f"{sample_df['subject'].nunique()} subjects "
        f"({sample_df['has_negation'].sum()} negation / "
        f"{(~sample_df['has_negation']).sum()} non-negation)."
    )
    if processed_ids:
        print(
            f"Found existing results at {out_path} — {len(processed_ids)} already done, "
            f"{len(remaining)} remaining. Resuming."
        )

    if len(remaining) == 0:
        print("Nothing left to evaluate — all questions already have results.")
    else:
        evaluator = LLMEvaluator(
            provider=args.provider,
            model_name=args.model,
            api_key=args.api_key,
            use_json=args.json,
        )

        for _, row in remaining.iterrows():
            options = [str(row["A"]), str(row["B"]), str(row["C"]), str(row["D"])]
            prompt = format_question(row["question"], options)
            raw_output = evaluator.query(prompt)
            pred_label = extract_answer(raw_output)
            is_correct = pred_label == row["label"]

            result_row = {
                "question_id": row["question_id"],
                "subject": row["subject"],
                "has_negation": row["has_negation"],
                "question": row["question"],
                "ground_truth": row["label"],
                "predicted": pred_label,
                "raw_output": raw_output,
                "correct": is_correct,
            }
            append_result_row(out_path, result_row, FIELDNAMES)

            print(
                f"[{row['subject']}] negation={row['has_negation']} | "
                f"GT={row['label']} | Pred={pred_label} | "
                f"{'Correct' if is_correct else 'Incorrect'}"
            )

    results_df = pd.read_csv(out_path)
    results_df["has_negation"] = results_df["has_negation"].astype(bool)
    results_df["correct"] = results_df["correct"].astype(bool)

    neg_results = results_df[results_df["has_negation"]]
    non_neg_results = results_df[~results_df["has_negation"]]

    neg_correct, neg_total = neg_results["correct"].sum(), len(neg_results)
    non_neg_correct, non_neg_total = (
        non_neg_results["correct"].sum(),
        len(non_neg_results),
    )

    neg_acc = (neg_correct / neg_total * 100) if neg_total else 0
    non_neg_acc = (non_neg_correct / non_neg_total * 100) if non_neg_total else 0

    z, p_value = two_proportion_z_test(
        neg_correct, neg_total, non_neg_correct, non_neg_total
    )

    print("\n=== OVERALL RESULTS ===")
    print(f"Negation questions:     {neg_acc:.2f}% ({neg_correct}/{neg_total})")
    print(f"Non-negation questions: {non_neg_acc:.2f}% ({non_neg_correct}/{non_neg_total})")
    if z is not None:
        print(
            f"Two-proportion z-test: z={z:.3f}, p={p_value:.4f} "
            f"({'significant at p<0.05' if p_value < 0.05 else 'not significant at p<0.05'})"
        )

    print("\n=== PER-SUBJECT BREAKDOWN ===")
    subject_summary = (
        results_df.groupby(["subject", "has_negation"])["correct"]
        .agg(["mean", "count"])
        .reset_index()
    )
    subject_summary["mean"] = (subject_summary["mean"] * 100).round(2)
    print(subject_summary.to_string(index=False))

    print(f"\nResults file (append-only, safe to resume): {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test whether negation in questions hurts LLM accuracy on MMLU"
    )
    add_llm_args(parser)
    parser.add_argument(
        "--subjects",
        type=str,
        default="all",
        help="Comma-separated subject list, or 'all' to scan every subject",
    )
    parser.add_argument(
        "--limit_per_group",
        type=int,
        default=20,
        help="Max questions sampled per negation/non-negation group per subject",
    )
    parser.add_argument(
        "--min_per_group",
        type=int,
        default=10,
        help="Minimum questions required in BOTH groups for a subject to be included",
    )
    parser.add_argument("--seed", type=int, default=42)

    run(parser.parse_args())
