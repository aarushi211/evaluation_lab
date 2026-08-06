"""
answer_only_baseline_eval.py

Check: Can a model beat random chance (25%) using ONLY the four options,
with the question text removed entirely?

If it can, that means the model is exploiting structural/phrasing cues in
the options themselves (length, specificity, "sounds like a distractor"
patterns) rather than reasoning about the actual question — a stronger,
more damning claim than length bias alone, since here there is nothing
correct to reason about at all.

Checkpointing
-------------
Each result is appended to the output CSV immediately. Rerunning the exact
same command auto-resumes by skipping already-completed question_ids.

Arguments
---------
  --provider   {ollama, groq, openai, anthropic, gemini}  (default: ollama)
  --model      Model name / id                            (default: llama3.2)
  --api_key    API key; comma-separated for rotation      (optional; else env)
  --json       Request JSON-shaped A/B/C/D answers        (flag)

  --subjects   Comma-separated subject list, or "all"     (default: all)
  --limit      Total questions to sample across subjects  (default: 100)
  --seed       Sampling RNG seed                          (default: 42)

Example
-------
  python answer_only_baseline_eval.py --provider groq --model llama-3.1-8b-instant \\
      --subjects all --limit 200 --json
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
    load_processed_ids,
    make_question_id,
    project_root,
)

FIELDNAMES = [
    "question_id",
    "subject",
    "options_only_prompt",
    "ground_truth",
    "predicted",
    "raw_output",
    "correct",
]


def format_options_only(options):
    prompt = "Question: [omitted]\n"
    prompt += f"A. {options[0]}\n"
    prompt += f"B. {options[1]}\n"
    prompt += f"C. {options[2]}\n"
    prompt += f"D. {options[3]}\n"
    prompt += (
        "Without seeing the question, which option is most likely to be correct? Answer:"
    )
    return prompt


def build_sample(data_dir: str, subjects, total_limit: int, seed: int) -> pd.DataFrame:
    all_rows = []
    for subject in subjects:
        test_file = os.path.join(data_dir, "test", f"{subject}_test.csv")
        if not os.path.exists(test_file):
            continue
        df = pd.read_csv(
            test_file, header=None, names=["question", "A", "B", "C", "D", "label"]
        )
        df["subject"] = subject
        df["label"] = df["label"].astype(str).str.strip()
        all_rows.append(df)

    if not all_rows:
        return pd.DataFrame()

    combined = pd.concat(all_rows, ignore_index=True)
    combined = combined[combined["label"].isin(["A", "B", "C", "D"])]
    sample = combined.sample(
        n=min(total_limit, len(combined)), random_state=seed
    ).reset_index(drop=True)
    sample["question_id"] = sample.apply(
        lambda r: make_question_id(r["subject"], r["question"]), axis=1
    )
    return sample


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
    data_dir = dataset_dir("MMLU", "data", "data", from_file=__file__)

    if args.subjects.strip().lower() == "all":
        test_dir = os.path.join(data_dir, "test")
        subjects = sorted(
            os.path.basename(f).replace("_test.csv", "")
            for f in glob.glob(os.path.join(test_dir, "*.csv"))
        )
    else:
        subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]

    sample_df = build_sample(data_dir, subjects, args.limit, args.seed)
    if sample_df.empty:
        print("No questions found for the given subjects.")
        return

    out_dir = os.path.join(project_root(__file__), "benchmarks", "MMLU", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_name = (
        f"answer_only_{args.provider}_{args.model.replace('/', '_').replace('.', '_')}"
        f"_lim{args.limit}.csv"
    )
    out_path = os.path.join(out_dir, out_name)

    processed_ids = load_processed_ids(out_path)
    remaining = sample_df[~sample_df["question_id"].isin(processed_ids)]

    print(
        f"Total sample: {len(sample_df)} questions across "
        f"{sample_df['subject'].nunique()} subjects. "
        f"Question text will be hidden from the model."
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
            prompt = format_options_only(options)
            raw_output = evaluator.query(prompt)
            pred_label = extract_answer(raw_output)
            is_correct = pred_label == row["label"]

            result_row = {
                "question_id": row["question_id"],
                "subject": row["subject"],
                "options_only_prompt": prompt.replace("\n", " | "),
                "ground_truth": row["label"],
                "predicted": pred_label,
                "raw_output": raw_output,
                "correct": is_correct,
            }
            append_result_row(out_path, result_row, FIELDNAMES)

            print(
                f"[{row['subject']}] GT={row['label']} | Pred={pred_label} | "
                f"{'Correct' if is_correct else 'Incorrect'}"
            )

    results_df = pd.read_csv(out_path)
    results_df["correct"] = results_df["correct"].astype(bool)

    total = len(results_df)
    correct = results_df["correct"].sum()
    acc = correct / total * 100 if total else 0
    z, p_value = one_sample_proportion_test(correct, total, null_p=0.25)

    print("\n=== OVERALL RESULTS ===")
    print(
        f"Options-only accuracy: {acc:.2f}% ({correct}/{total})  [random chance = 25%]"
    )
    if z is not None:
        print(
            f"One-sample proportion test vs. 25% chance: z={z:.3f}, p={p_value:.4f} "
            f"({'SIGNIFICANTLY above/below chance' if p_value < 0.05 else 'not significantly different from chance'})"
        )

    print("\n=== PER-SUBJECT BREAKDOWN ===")
    subj_summary = (
        results_df.groupby("subject")["correct"].agg(["mean", "count"]).reset_index()
    )
    subj_summary["mean"] = (subj_summary["mean"] * 100).round(2)
    subj_summary = subj_summary.sort_values("mean", ascending=False)
    print(subj_summary.to_string(index=False))

    print(f"\nResults file (append-only, safe to resume): {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test whether a model can beat chance using ONLY the options, question text hidden"
    )
    add_llm_args(parser)
    parser.add_argument(
        "--subjects",
        type=str,
        default="all",
        help="Comma-separated subject list, or 'all' to sample across every subject",
    )
    parser.add_argument(
        "--limit", type=int, default=100, help="Total number of questions to sample"
    )
    parser.add_argument("--seed", type=int, default=42)

    run(parser.parse_args())
