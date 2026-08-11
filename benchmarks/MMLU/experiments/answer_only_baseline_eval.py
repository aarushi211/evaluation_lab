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
same command auto-resumes by skipping already-completed row_ids.
question_id = MD5(subject::question); row_id = MD5(subject::row_number)
so duplicate stems are still evaluated.

CSV columns (length / prompt metadata)
--------------------------------------
  prompt_template           Fixed instruction text (options filled in at eval time)
  option_A .. option_D      Raw option text
  A_length .. D_length      Character lengths of each option
  longest_option            Letter(s) with max length (pipe-joined on ties, e.g. "B|C")
  longest_option_length     That max character length
  correct_is_longest        True if ground-truth is among the longest option(s)
  predicted_is_longest      True if the model's prediction is among the longest
  predicted_option_length   Character length of the predicted option ("" if N/A)
  predicted_rank_by_length  1 = longest, 2 = second-longest, … ("" if N/A; ties share rank)

Arguments
---------
  --provider   {ollama, groq, openai, anthropic, gemini}  (default: ollama)
  --model      Model name / id                            (default: llama3.2)
  --api_key    API key; comma-separated for rotation      (optional; else env)
  --json       Request JSON-shaped A/B/C/D answers        (flag)

  --subjects   Comma-separated subject list, or "all"     (default: all)
  --limit      Total questions to sample across subjects  (default: 100)
  --seed       Sampling RNG seed                          (default: 42)
  --data_dir   Folder containing test/ — Colab-friendly
               (default: <repo>/datasets/MMLU/data/data)
  --workers    Concurrent API threads (default: 1; try 4–8 for cloud APIs)

Example
-------
  python answer_only_baseline_eval.py --provider groq --model llama-3.1-8b-instant \\
      --subjects all --limit 200 --json --workers 8

  python answer_only_baseline_eval.py --data_dir /content/data --limit 50
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
    add_data_dir_arg,
    add_llm_args,
    add_workers_arg,
    append_result_row,
    extract_answer,
    assign_eval_ids,
    filter_unprocessed,
    project_root,
    resolve_data_dir,
    run_parallel,
)

PROMPT_TEMPLATE = (
    "Question: [omitted]\n"
    "A. {option_A}\n"
    "B. {option_B}\n"
    "C. {option_C}\n"
    "D. {option_D}\n"
    "\n"
    "The original question is unavailable.\n"
    "\n"
    "Below are the four answer options.\n"
    "\n"
    "Choose the option that is most likely to be correct.\n"
    "\n"
    "Respond with only A, B, C or D."
)

FIELDNAMES = [
    "question_id",
    "row_id",
    "subject",
    "prompt_template",
    "option_A",
    "option_B",
    "option_C",
    "option_D",
    "A_length",
    "B_length",
    "C_length",
    "D_length",
    "longest_option",
    "longest_option_length",
    "correct_is_longest",
    "predicted_is_longest",
    "predicted_option_length",
    "predicted_rank_by_length",
    "ground_truth",
    "predicted",
    "raw_output",
    "correct",
]


def format_options_only(options):
    """Build the API prompt from the fixed template + four option strings."""
    return PROMPT_TEMPLATE.format(
        option_A=options[0],
        option_B=options[1],
        option_C=options[2],
        option_D=options[3],
    )


def option_length_features(options, correct_label: str):
    """
    Compute length-bias features for the four options.

    Returns
    -------
    length_by_label : dict
        {"A": n, "B": n, "C": n, "D": n}
    longest_option : str
        Letter(s) with max character length; pipe-joined on ties
    longest_option_length : int
        That max length
    correct_is_longest : bool
        Whether the ground-truth label is among the longest option(s)
    rank_by_label : dict
        Dense rank by length (1 = longest). Ties share the same rank.
    """
    labels = ["A", "B", "C", "D"]
    lengths = [len(str(opt)) for opt in options]
    length_by_label = dict(zip(labels, lengths))
    max_len = max(lengths) if lengths else 0
    longest = [lab for lab, n in zip(labels, lengths) if n == max_len]
    longest_option = "|".join(longest)
    correct_is_longest = str(correct_label).strip().upper() in longest

    # Dense rank: unique lengths sorted descending → 1, 2, 3, …
    unique_desc = sorted(set(lengths), reverse=True)
    rank_of_length = {n: i + 1 for i, n in enumerate(unique_desc)}
    rank_by_label = {lab: rank_of_length[n] for lab, n in length_by_label.items()}

    return (
        length_by_label,
        longest_option,
        max_len,
        correct_is_longest,
        rank_by_label,
    )


def predicted_length_features(
    predicted_label: str,
    longest_option: str,
    length_by_label: dict,
    rank_by_label: dict,
):
    """
    Return (predicted_is_longest, predicted_option_length, predicted_rank_by_length).

    Length/rank are "" when the prediction is not a valid A–D letter.
    """
    pred = str(predicted_label).strip().upper()
    if pred not in length_by_label:
        return False, "", ""
    longest_set = set(str(longest_option).split("|")) if longest_option else set()
    return pred in longest_set, length_by_label[pred], rank_by_label[pred]


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
        df["row_no"] = df.index + 1
        df["label"] = df["label"].astype(str).str.strip()
        all_rows.append(df)

    if not all_rows:
        return pd.DataFrame()

    combined = pd.concat(all_rows, ignore_index=True)
    combined = combined[combined["label"].isin(["A", "B", "C", "D"])]
    sample = combined.sample(
        n=min(total_limit, len(combined)), random_state=seed
    ).reset_index(drop=True)
    sample = assign_eval_ids(sample)
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
    data_dir = resolve_data_dir(args.data_dir, from_file=__file__)
    print(f"Using data_dir: {data_dir}")

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

    remaining = filter_unprocessed(sample_df, out_path)

    print(
        f"Total sample: {len(sample_df)} questions across "
        f"{sample_df['subject'].nunique()} subjects. "
        f"Question text will be hidden from the model."
    )
    already = len(sample_df) - len(remaining)
    if already:
        print(
            f"Found existing results at {out_path} — {already} already done, "
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

        work_items = list(remaining.iterrows())

        def process_one(item):
            _, row = item
            options = [str(row["A"]), str(row["B"]), str(row["C"]), str(row["D"])]
            (
                length_by_label,
                longest_option,
                longest_option_length,
                correct_is_longest,
                rank_by_label,
            ) = option_length_features(options, row["label"])
            prompt = format_options_only(options)
            raw_output = evaluator.query(prompt)
            pred_label = extract_answer(raw_output)
            is_correct = pred_label == row["label"]
            (
                predicted_is_longest,
                predicted_option_length,
                predicted_rank_by_length,
            ) = predicted_length_features(
                pred_label, longest_option, length_by_label, rank_by_label
            )

            result_row = {
                "question_id": row["question_id"],
                "row_id": row["row_id"],
                "subject": row["subject"],
                "prompt_template": PROMPT_TEMPLATE.replace("\n", " | "),
                "option_A": options[0],
                "option_B": options[1],
                "option_C": options[2],
                "option_D": options[3],
                "A_length": length_by_label["A"],
                "B_length": length_by_label["B"],
                "C_length": length_by_label["C"],
                "D_length": length_by_label["D"],
                "longest_option": longest_option,
                "longest_option_length": longest_option_length,
                "correct_is_longest": correct_is_longest,
                "predicted_is_longest": predicted_is_longest,
                "predicted_option_length": predicted_option_length,
                "predicted_rank_by_length": predicted_rank_by_length,
                "ground_truth": row["label"],
                "predicted": pred_label,
                "raw_output": raw_output,
                "correct": is_correct,
            }
            append_result_row(out_path, result_row, FIELDNAMES)
            print(
                f"[{row['subject']}] GT={row['label']} | Pred={pred_label} | "
                f"longest={longest_option}({longest_option_length}) "
                f"pred_len={predicted_option_length} "
                f"pred_rank={predicted_rank_by_length} | "
                f"{'Correct' if is_correct else 'Incorrect'}"
            )
            return result_row

        print(f"Running with {max(1, int(args.workers))} worker(s)...")
        run_parallel(process_one, work_items, workers=args.workers)

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

    if "correct_is_longest" in results_df.columns:
        results_df["correct_is_longest"] = results_df["correct_is_longest"].astype(bool)
        longest_rate = results_df["correct_is_longest"].mean() * 100
        print(
            f"Ground truth is longest option: {longest_rate:.2f}% of questions "
            f"({results_df['correct_is_longest'].sum()}/{total})"
        )

    if "predicted_is_longest" in results_df.columns:
        results_df["predicted_is_longest"] = results_df["predicted_is_longest"].astype(bool)
        pick_longest_rate = results_df["predicted_is_longest"].mean() * 100
        print(
            f"Model predicted a longest option: {pick_longest_rate:.2f}% "
            f"({results_df['predicted_is_longest'].sum()}/{total})"
        )

    if "predicted_option_length" in results_df.columns:
        lengths = pd.to_numeric(results_df["predicted_option_length"], errors="coerce")
        if lengths.notna().any():
            print(
                f"Mean predicted option length: {lengths.mean():.1f} chars "
                f"(median={lengths.median():.1f})"
            )

    if "predicted_rank_by_length" in results_df.columns:
        ranks = pd.to_numeric(results_df["predicted_rank_by_length"], errors="coerce")
        if ranks.notna().any():
            top2 = (ranks <= 2).sum()
            print(
                f"Predicted longest or 2nd-longest: {top2 / ranks.notna().sum() * 100:.2f}% "
                f"({top2}/{ranks.notna().sum()})"
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
    add_data_dir_arg(parser)
    add_workers_arg(parser)
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
