"""
evaluate_mmlu.py

Standard MMLU subject evaluation (zero-/few-shot, optional option shuffle).

Uses utilities.LLMEvaluator so the same run works with ollama / groq / openai /
anthropic / gemini without changing this script.

Checkpointing
-------------
Each question is appended to the results CSV immediately (flush + fsync).
question_id is a stable MD5 of "subject::question". Rerun the exact same
command after a rate-limit / crash to skip already-completed IDs and continue.

Output
------
  benchmarks/MMLU/results/
    {provider}_{model}_{subject}_{shuffled|no_shuffle}_{N}shot_lim{L}.csv

Arguments
---------
  --provider   {ollama, groq, openai, anthropic, gemini}  (default: ollama)
  --model      Model name / id                            (default: llama3.2)
  --api_key    API key; comma-separated for rotation      (optional; else env)
  --json       Request JSON-shaped A/B/C/D answers        (flag)

  --subject    MMLU subject slug                          (default: anatomy)
  --shots      Number of few-shot examples from the dev split (default: 0)
  --shuffle    Shuffle A–D option order (deterministic per row index) (flag)
  --limit      Max test questions to evaluate             (default: 10)
  --data_dir   Folder containing test/ (and optionally dev/) — Colab-friendly
               (default: <repo>/datasets/MMLU/data/data)
  --workers    Concurrent API threads (default: 1; try 4–8 for cloud APIs)

Example
-------
  python evaluate_mmlu.py --provider openai --model gpt-4o-mini \\
      --subject global_facts --shots 5 --limit 100 --json --workers 8

  # Colab / custom layout:
  python evaluate_mmlu.py --data_dir /content/data --subject anatomy --limit 10
"""

import os
import sys
import random
import argparse
import pandas as pd

# Repo root must be on sys.path before importing utilities/
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
    format_question,
    generate_few_shot_prefix,
    load_processed_ids,
    make_question_id,
    project_root,
    resolve_data_dir,
    run_parallel,
)

FIELDNAMES = [
    "question_id",
    "question_no",
    "ground_truth",
    "predicted",
    "correct",
    "raw_output",
    "provider",
    "model",
    "subject",
    "shots",
    "shuffle",
    "limit",
]


def run_evaluation(
    data_dir: str,
    subject: str,
    evaluator: LLMEvaluator,
    num_shots: int = 0,
    shuffle_options: bool = False,
    limit: int = None,
    workers: int = 1,
):
    print(
        f"\n--- Evaluating Subject: {subject} "
        f"(Shuffled Options: {shuffle_options}, Shots: {num_shots}, Workers: {workers}) ---"
    )

    test_file = os.path.join(data_dir, "test", f"{subject}_test.csv")
    dev_file = os.path.join(data_dir, "dev", f"{subject}_dev.csv")

    if not os.path.exists(test_file):
        print(f"Test file not found for {subject}")
        return

    test_df = pd.read_csv(
        test_file, header=None, names=["question", "A", "B", "C", "D", "label"]
    )

    few_shot_prefix = ""
    if num_shots > 0 and os.path.exists(dev_file):
        dev_df = pd.read_csv(
            dev_file, header=None, names=["question", "A", "B", "C", "D", "label"]
        )
        few_shot_prefix = generate_few_shot_prefix(dev_df, num_shots)

    if limit:
        test_df = test_df.head(limit)

    # Stable IDs for resume; keep original row index for deterministic shuffle seeds.
    test_df = test_df.copy()
    test_df["question_id"] = test_df["question"].apply(
        lambda q: make_question_id(subject, q)
    )
    test_df["question_no"] = range(1, len(test_df) + 1)

    out_dir = os.path.join(project_root(__file__), "benchmarks", "MMLU", "results")
    os.makedirs(out_dir, exist_ok=True)
    model_slug = evaluator.model_name.replace("/", "_").replace(".", "_")
    shuffle_tag = "shuffled" if shuffle_options else "no_shuffle"
    lim_tag = limit if limit is not None else "all"
    out_name = (
        f"{evaluator.provider}_{model_slug}_{subject}_{shuffle_tag}_"
        f"{num_shots}shot_lim{lim_tag}.csv"
    )
    out_path = os.path.join(out_dir, out_name)

    processed_ids = load_processed_ids(out_path)
    remaining = test_df[~test_df["question_id"].isin(processed_ids)]

    print(f"Total sample: {len(test_df)} questions.")
    if processed_ids:
        print(
            f"Found existing results at {out_path} — {len(processed_ids)} already done, "
            f"{len(remaining)} remaining. Resuming."
        )

    if len(remaining) == 0:
        print("Nothing left to evaluate — all questions already have results.")
    else:
        work_items = list(remaining.iterrows())

        def process_one(item):
            idx, row = item
            question = row["question"]
            original_options = [
                str(row["A"]),
                str(row["B"]),
                str(row["C"]),
                str(row["D"]),
            ]
            original_label = str(row["label"]).strip()

            if shuffle_options:
                indexed_options = list(zip(["A", "B", "C", "D"], original_options))
                random.seed(idx)
                random.shuffle(indexed_options)

                shuffled_options = [opt[1] for opt in indexed_options]
                new_label = None
                for new_idx, (orig_letter, _) in enumerate(indexed_options):
                    if orig_letter == original_label:
                        new_label = ["A", "B", "C", "D"][new_idx]
                        break
            else:
                shuffled_options = original_options
                new_label = original_label

            prompt = few_shot_prefix + format_question(question, shuffled_options)
            raw_output = evaluator.query(prompt)
            pred_label = extract_answer(raw_output)
            is_correct = pred_label == new_label

            result_row = {
                "question_id": row["question_id"],
                "question_no": int(row["question_no"]),
                "ground_truth": new_label,
                "predicted": pred_label,
                "correct": is_correct,
                "raw_output": raw_output,
                "provider": evaluator.provider,
                "model": evaluator.model_name,
                "subject": subject,
                "shots": num_shots,
                "shuffle": shuffle_options,
                "limit": limit if limit is not None else "",
            }
            append_result_row(out_path, result_row, FIELDNAMES)
            print(
                f"Q{row['question_no']}: GroundTruth={new_label} | Predicted={pred_label} | "
                f"RawOutput='{raw_output.strip()}' | "
                f"{'Correct' if is_correct else 'Incorrect'}"
            )
            return result_row

        print(f"Running with {max(1, int(workers))} worker(s)...")
        run_parallel(process_one, work_items, workers=workers)

    results_df = pd.read_csv(out_path)
    results_df["correct"] = results_df["correct"].astype(bool)
    correct_count = int(results_df["correct"].sum())
    total_count = len(results_df)
    accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
    print(f"Accuracy: {accuracy:.2f}% ({correct_count}/{total_count})")
    print(f"Results file (append-only, safe to resume): {out_path}")
    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate MMLU subjects using ollama / groq / openai / anthropic / gemini"
    )
    add_llm_args(parser)
    add_data_dir_arg(parser)
    add_workers_arg(parser)
    parser.add_argument("--subject", type=str, default="anatomy", help="MMLU subject to run")
    parser.add_argument("--shots", type=int, default=0, help="Number of few-shot examples")
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle options to test position/length sensitivity",
    )
    parser.add_argument(
        "--limit", type=int, default=10, help="Limit number of questions evaluated"
    )

    args = parser.parse_args()
    data_dir = resolve_data_dir(args.data_dir, from_file=__file__)
    print(f"Using data_dir: {data_dir}")

    try:
        evaluator = LLMEvaluator(
            provider=args.provider,
            model_name=args.model,
            api_key=args.api_key,
            use_json=args.json,
        )
        run_evaluation(
            data_dir=data_dir,
            subject=args.subject,
            evaluator=evaluator,
            num_shots=args.shots,
            shuffle_options=args.shuffle,
            limit=args.limit,
            workers=args.workers,
        )
    except Exception as e:
        print(f"Initialization/Execution error: {e}")
