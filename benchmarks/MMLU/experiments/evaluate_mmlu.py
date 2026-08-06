"""
evaluate_mmlu.py

Standard MMLU subject evaluation (zero-/few-shot, optional option shuffle).

Uses utilities.LLMEvaluator so the same run works with ollama / groq / openai /
anthropic / gemini without changing this script.

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

Example
-------
  python evaluate_mmlu.py --provider openai --model gpt-4o-mini \\
      --subject global_facts --shots 5 --limit 100 --json
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
    add_llm_args,
    dataset_dir,
    extract_answer,
    format_question,
    generate_few_shot_prefix,
)


def run_evaluation(
    data_dir: str,
    subject: str,
    evaluator: LLMEvaluator,
    num_shots: int = 0,
    shuffle_options: bool = False,
    limit: int = None,
):
    print(
        f"\n--- Evaluating Subject: {subject} "
        f"(Shuffled Options: {shuffle_options}, Shots: {num_shots}) ---"
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

    correct_count = 0
    total_count = 0

    for idx, row in test_df.iterrows():
        question = row["question"]
        original_options = [str(row["A"]), str(row["B"]), str(row["C"]), str(row["D"])]
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
        if is_correct:
            correct_count += 1
        total_count += 1

        print(
            f"Q{idx + 1}: GroundTruth={new_label} | Predicted={pred_label} | "
            f"RawOutput='{raw_output.strip()}' | "
            f"{'Correct' if is_correct else 'Incorrect'}"
        )

    accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
    print(f"Accuracy: {accuracy:.2f}% ({correct_count}/{total_count})")
    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate MMLU subjects using ollama / groq / openai / anthropic / gemini"
    )
    add_llm_args(parser)
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
    data_dir = dataset_dir("MMLU", "data", "data", from_file=__file__)

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
        )
    except Exception as e:
        print(f"Initialization/Execution error: {e}")
