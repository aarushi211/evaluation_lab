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
    Single subject:
      {provider}_{model}_{subject}_{shuffled|no_shuffle}_{N}shot_lim{L}.csv
    Multi-subject / --subjects all (one shared file):
      {provider}_{model}_all_{shuffled|no_shuffle}_{N}shot_lim{L}.csv

Arguments
---------
  --provider   {ollama, groq, openai, anthropic, gemini}  (default: ollama)
  --model      Model name / id                            (default: llama3.2)
  --api_key    API key; comma-separated for rotation      (optional; else env)
  --json       Request JSON-shaped A/B/C/D answers        (flag)

  --subjects   Comma-separated subject list, or "all" for every test subject
               (default: anatomy). --subject is accepted as an alias.
  --shots      Number of few-shot examples from the dev split (default: 0)
  --shuffle    Shuffle A–D option order (deterministic per row index) (flag)
  --limit      Max test questions per subject             (default: 10;
               use 0 for no limit / all questions in each subject)
  --data_dir   Folder containing test/ (and optionally dev/) — Colab-friendly
               (default: <repo>/datasets/MMLU/data/data)
  --workers    Concurrent API threads (default: 1; try 4–8 for cloud APIs)

Example
-------
  # Single subject, 10 questions (default smoke test)
  python evaluate_mmlu.py --provider openai --model gpt-4o-mini --json

  # Full MMLU test set (all subjects, all questions) → one shared CSV
  python evaluate_mmlu.py --provider openai --model gpt-5-nano --json \\
      --subjects all --limit 0 --workers 8

  # Several subjects, capped → one shared CSV
  python evaluate_mmlu.py --subjects anatomy,global_facts --limit 50 --json
"""

import os
import sys
import glob
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


def list_subjects(data_dir: str):
    """Sorted subject slugs that have a test CSV under data_dir/test/."""
    test_dir = os.path.join(data_dir, "test")
    return sorted(
        os.path.basename(f).replace("_test.csv", "")
        for f in glob.glob(os.path.join(test_dir, "*.csv"))
    )


def parse_subjects(subjects_arg: str, data_dir: str):
    """Expand 'all' or a comma-separated list into subject slugs."""
    raw = subjects_arg.strip()
    if raw.lower() == "all":
        subjects = list_subjects(data_dir)
        if not subjects:
            raise FileNotFoundError(f"No test CSVs found under {os.path.join(data_dir, 'test')}")
        return subjects
    subjects = [s.strip() for s in raw.split(",") if s.strip()]
    if not subjects:
        raise ValueError("No subjects provided. Pass --subjects all or a comma-separated list.")
    return subjects


def results_out_path(
    evaluator: LLMEvaluator,
    subjects: list,
    shuffle_options: bool,
    num_shots: int,
    limit: int = None,
) -> str:
    """One shared CSV for multi-subject / all; subject-named CSV for a single subject."""
    out_dir = os.path.join(project_root(__file__), "benchmarks", "MMLU", "results")
    os.makedirs(out_dir, exist_ok=True)
    model_slug = evaluator.model_name.replace("/", "_").replace(".", "_")
    shuffle_tag = "shuffled" if shuffle_options else "no_shuffle"
    lim_tag = limit if limit is not None and limit > 0 else "all"
    scope = "all" if len(subjects) > 1 else subjects[0]
    out_name = (
        f"{evaluator.provider}_{model_slug}_{scope}_{shuffle_tag}_"
        f"{num_shots}shot_lim{lim_tag}.csv"
    )
    return os.path.join(out_dir, out_name)


def run_evaluation(
    data_dir: str,
    subject: str,
    evaluator: LLMEvaluator,
    out_path: str,
    num_shots: int = 0,
    shuffle_options: bool = False,
    limit: int = None,
    workers: int = 1,
):
    lim_tag = limit if limit is not None and limit > 0 else "all"
    print(
        f"\n--- Evaluating Subject: {subject} "
        f"(Shuffled Options: {shuffle_options}, Shots: {num_shots}, "
        f"Limit: {lim_tag}, Workers: {workers}) ---"
    )

    test_file = os.path.join(data_dir, "test", f"{subject}_test.csv")
    dev_file = os.path.join(data_dir, "dev", f"{subject}_dev.csv")

    if not os.path.exists(test_file):
        print(f"Test file not found for {subject}")
        return None

    test_df = pd.read_csv(
        test_file, header=None, names=["question", "A", "B", "C", "D", "label"]
    )

    few_shot_prefix = ""
    if num_shots > 0 and os.path.exists(dev_file):
        dev_df = pd.read_csv(
            dev_file, header=None, names=["question", "A", "B", "C", "D", "label"]
        )
        few_shot_prefix = generate_few_shot_prefix(dev_df, num_shots)

    if limit is not None and limit > 0:
        test_df = test_df.head(limit)

    # Stable IDs for resume; keep original row index for deterministic shuffle seeds.
    test_df = test_df.copy()
    test_df["question_id"] = test_df["question"].apply(
        lambda q: make_question_id(subject, q)
    )
    test_df["question_no"] = range(1, len(test_df) + 1)

    processed_ids = load_processed_ids(out_path)
    remaining = test_df[~test_df["question_id"].isin(processed_ids)]

    print(f"Total sample: {len(test_df)} questions.")
    if processed_ids:
        already = len(test_df) - len(remaining)
        print(
            f"Shared results at {out_path} — {already}/{len(test_df)} for this subject "
            f"already done, {len(remaining)} remaining. Resuming."
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
                "limit": lim_tag,
            }
            append_result_row(out_path, result_row, FIELDNAMES)
            print(
                f"[{subject}] Q{row['question_no']}: GroundTruth={new_label} | "
                f"Predicted={pred_label} | RawOutput='{raw_output.strip()}' | "
                f"{'Correct' if is_correct else 'Incorrect'}"
            )
            return result_row

        print(f"Running with {max(1, int(workers))} worker(s)...")
        run_parallel(process_one, work_items, workers=workers)

    if not os.path.exists(out_path):
        return None

    results_df = pd.read_csv(out_path)
    subject_df = results_df[results_df["subject"] == subject]
    if len(subject_df) == 0:
        return None
    subject_df = subject_df.copy()
    subject_df["correct"] = subject_df["correct"].astype(bool)
    correct_count = int(subject_df["correct"].sum())
    total_count = len(subject_df)
    accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
    print(f"Accuracy [{subject}]: {accuracy:.2f}% ({correct_count}/{total_count})")
    return {
        "subject": subject,
        "accuracy": accuracy,
        "correct": correct_count,
        "total": total_count,
        "out_path": out_path,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate MMLU subjects using ollama / groq / openai / anthropic / gemini"
    )
    add_llm_args(parser)
    add_data_dir_arg(parser)
    add_workers_arg(parser)
    parser.add_argument(
        "--subjects",
        "--subject",
        dest="subjects",
        type=str,
        default="anatomy",
        help="Comma-separated subject list, or 'all' for every test subject (default: anatomy)",
    )
    parser.add_argument("--shots", type=int, default=0, help="Number of few-shot examples")
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle options to test position/length sensitivity",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Max questions per subject (default: 10; use 0 for no limit / all questions)",
    )

    args = parser.parse_args()
    data_dir = resolve_data_dir(args.data_dir, from_file=__file__)
    print(f"Using data_dir: {data_dir}")

    # 0 => no per-subject cap (full subject / full dataset when combined with --subjects all)
    limit = None if args.limit == 0 else args.limit

    try:
        subjects = parse_subjects(args.subjects, data_dir)
        print(
            f"Subjects ({len(subjects)}): {', '.join(subjects[:8])}"
            + ("..." if len(subjects) > 8 else "")
        )

        evaluator = LLMEvaluator(
            provider=args.provider,
            model_name=args.model,
            api_key=args.api_key,
            use_json=args.json,
        )

        out_path = results_out_path(
            evaluator=evaluator,
            subjects=subjects,
            shuffle_options=args.shuffle,
            num_shots=args.shots,
            limit=limit,
        )
        print(f"Results file (append-only, safe to resume): {out_path}")

        summaries = []
        for subject in subjects:
            summary = run_evaluation(
                data_dir=data_dir,
                subject=subject,
                evaluator=evaluator,
                out_path=out_path,
                num_shots=args.shots,
                shuffle_options=args.shuffle,
                limit=limit,
                workers=args.workers,
            )
            if summary is not None:
                summaries.append(summary)

        if len(summaries) > 1:
            total_correct = sum(s["correct"] for s in summaries)
            total_n = sum(s["total"] for s in summaries)
            overall = (total_correct / total_n * 100) if total_n else 0
            print("\n=== OVERALL (all subjects) ===")
            print(f"Accuracy: {overall:.2f}% ({total_correct}/{total_n})")
            print(f"Results file: {out_path}")
            print("\n=== PER-SUBJECT ===")
            summary_df = pd.DataFrame(summaries)[["subject", "accuracy", "correct", "total"]]
            summary_df = summary_df.sort_values("accuracy", ascending=False)
            print(summary_df.to_string(index=False))
    except Exception as e:
        print(f"Initialization/Execution error: {e}")
