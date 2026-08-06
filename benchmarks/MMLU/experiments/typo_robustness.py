"""
typo_robustness.py

Research Question 5: Do typos in questions affect LLM performance?

Methodology
-----------
This is a PAIRED experiment: the same question is asked twice, once with the
original text and once with a synthetically typo'd version (options and
ground-truth label are left untouched). Because every question is its own
control, we can isolate the effect of typos from question difficulty and use
McNemar's test on the paired outcomes.

Typo generation (seeded per-question for reproducibility) randomly applies one
of four realistic error types to a subset of words in the question text:
  - adjacent-key substitution (QWERTY keyboard adjacency)
  - character deletion
  - character duplication
  - adjacent character transposition

Only the question stem is perturbed; options (A-D) are left clean so we're
testing reading robustness, not making the options unrecognizable.

Checkpointing
-------------
Each question is evaluated as a pair (original + typo'd) and appended to the
CSV immediately (2 API calls per row). Rerun the same command to resume.
If a run dies mid-pair, that pair is re-run from scratch on resume.

Arguments
---------
  --provider   {ollama, groq, openai, anthropic, gemini}  (default: ollama)
  --model      Model name / id                            (default: llama3.2)
  --api_key    API key; comma-separated for rotation      (optional; else env)
  --json       Request JSON-shaped A/B/C/D answers        (flag)

  --subjects   Comma-separated subject list, or "all"     (default: all)
  --limit      Total questions to sample                  (default: 100)
  --typo_rate  Fraction of eligible words to perturb      (default: 0.15)
  --seed       Sampling / typo RNG seed                   (default: 42)

Example
-------
  python typo_robustness.py --provider gemini --model gemini-2.0-flash \\
      --subjects all --limit 100 --typo_rate 0.15 --json
"""

import os
import sys
import glob
import random
import argparse
import re
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

QWERTY_ADJACENCY = {
    "q": "wa",
    "w": "qes",
    "e": "wrd",
    "r": "etf",
    "t": "ryg",
    "y": "tuh",
    "u": "yij",
    "i": "uok",
    "o": "ipl",
    "p": "ol",
    "a": "qsz",
    "s": "awedz",
    "d": "sefc",
    "f": "drgv",
    "g": "ftyb",
    "h": "gyun",
    "j": "huim",
    "k": "jiol",
    "l": "kop",
    "z": "asx",
    "x": "zsdc",
    "c": "xdfv",
    "v": "cfgb",
    "b": "vghn",
    "n": "bhjm",
    "m": "njk",
}

FIELDNAMES = [
    "question_id",
    "subject",
    "question",
    "typo_question",
    "ground_truth",
    "orig_pred",
    "orig_correct",
    "typo_pred",
    "typo_correct",
]


def introduce_typo_in_word(word: str, rng: random.Random) -> str:
    """
    Return a modified version of `word` that is guaranteed to differ from the original
    whenever len(word) > 1.
    """
    if len(word) < 2:
        return word

    ops = ["substitute", "delete", "duplicate", "transpose"]
    rng.shuffle(ops)
    lower = word.lower()

    for op in ops:
        if op == "substitute":
            candidates = [i for i, ch in enumerate(lower) if ch in QWERTY_ADJACENCY]
            if candidates:
                pos = rng.choice(candidates)
                replacement = rng.choice(QWERTY_ADJACENCY[lower[pos]])
                mutated = word[:pos] + replacement + word[pos + 1 :]
                if mutated != word:
                    return mutated

        elif op == "delete":
            pos = rng.randint(0, len(word) - 1)
            mutated = word[:pos] + word[pos + 1 :]
            if mutated != word:
                return mutated

        elif op == "duplicate":
            pos = rng.randint(0, len(word) - 1)
            mutated = word[:pos] + word[pos] + word[pos:]
            if mutated != word:
                return mutated

        elif op == "transpose" and len(word) >= 2:
            pos = rng.randint(0, len(word) - 2)
            chars = list(word)
            chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
            mutated = "".join(chars)
            if mutated != word:
                return mutated

    # Guaranteed fallback: delete one character.
    pos = rng.randint(0, len(word) - 1)
    mutated = word[:pos] + word[pos + 1 :]
    if mutated != word:
        return mutated

    # Final fallback for pathological cases.
    return word + "x"


def introduce_typos(text: str, rate: float, seed: int) -> str:
    rng = random.Random(seed)
    words = text.split(" ")

    eligible_idx = [
        i for i, w in enumerate(words) if len(re.sub(r"[^a-zA-Z]", "", w)) >= 3
    ]

    if not eligible_idx:
        return text

    num_to_perturb = max(1, round(len(eligible_idx) * rate))
    num_to_perturb = min(num_to_perturb, len(eligible_idx))

    chosen = rng.sample(eligible_idx, k=num_to_perturb)

    original_text = text
    for i in chosen:
        words[i] = introduce_typo_in_word(words[i], rng)

    typo_text = " ".join(words)

    # Safety net: if somehow nothing changed, force one guaranteed edit.
    if typo_text == original_text:
        i = rng.choice(eligible_idx)
        words[i] = words[i][:-1] if len(words[i]) > 1 else words[i] + "x"
        typo_text = " ".join(words)

    return typo_text

def mcnemar_test(b: int, c: int):
    """
    Paired binary comparison.
    b = correct on original, incorrect on typo
    c = incorrect on original, correct on typo
    Uses continuity-corrected chi-square approximation (valid when b + c >= ~10;
    with smaller samples treat the p-value as indicative only).
    """
    if b + c == 0:
        return 0.0, 1.0
    chi2 = ((abs(b - c) - 1) ** 2) / (b + c)
    p_value = math.erfc(math.sqrt(chi2 / 2))
    return chi2, p_value


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
    sample = combined.sample(
        n=min(total_limit, len(combined)), random_state=seed
    ).reset_index(drop=True)
    sample["question_id"] = sample.apply(
        lambda r: make_question_id(r["subject"], r["question"]), axis=1
    )
    return sample


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

    sample_df["typo_question"] = [
        introduce_typos(q, args.typo_rate, seed=args.seed + i)
        for i, q in enumerate(sample_df["question"])
    ]

    out_dir = os.path.join(project_root(__file__), "benchmarks", "MMLU", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_name = (
        f"typos_{args.provider}_{args.model.replace('/', '_').replace('.', '_')}"
        f"_lim{args.limit}.csv"
    )
    out_path = os.path.join(out_dir, out_name)

    processed_ids = load_processed_ids(out_path)
    remaining = sample_df[~sample_df["question_id"].isin(processed_ids)]

    print(
        f"Total sample: {len(sample_df)} questions across "
        f"{sample_df['subject'].nunique()} subjects. "
        f"Each requires 2 API calls (original + typo'd)."
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
            gt = row["label"]

            orig_prompt = format_question(row["question"], options)
            orig_raw = evaluator.query(orig_prompt)
            orig_pred = extract_answer(orig_raw)
            orig_correct = orig_pred == gt

            typo_prompt = format_question(row["typo_question"], options)
            typo_raw = evaluator.query(typo_prompt)
            typo_pred = extract_answer(typo_raw)
            typo_correct = typo_pred == gt

            result_row = {
                "question_id": row["question_id"],
                "subject": row["subject"],
                "question": row["question"],
                "typo_question": row["typo_question"],
                "ground_truth": gt,
                "orig_pred": orig_pred,
                "orig_correct": orig_correct,
                "typo_pred": typo_pred,
                "typo_correct": typo_correct,
            }
            append_result_row(out_path, result_row, FIELDNAMES)

            print(
                f"[{row['subject']}] GT={gt} | "
                f"Orig={orig_pred} ({'OK' if orig_correct else 'X'}) | "
                f"Typo={typo_pred} ({'OK' if typo_correct else 'X'})"
            )

    results_df = pd.read_csv(out_path)
    results_df["orig_correct"] = results_df["orig_correct"].astype(bool)
    results_df["typo_correct"] = results_df["typo_correct"].astype(bool)

    orig_acc = results_df["orig_correct"].mean() * 100
    typo_acc = results_df["typo_correct"].mean() * 100

    b = ((results_df["orig_correct"]) & (~results_df["typo_correct"])).sum()
    c = ((~results_df["orig_correct"]) & (results_df["typo_correct"])).sum()
    chi2, p_value = mcnemar_test(b, c)

    print("\n=== OVERALL RESULTS ===")
    print(f"Original accuracy: {orig_acc:.2f}%")
    print(f"Typo'd accuracy:   {typo_acc:.2f}%")
    print(
        f"Flipped correct->incorrect due to typo: {b} | Flipped incorrect->correct: {c}"
    )
    print(
        f"McNemar's test: chi2={chi2:.3f}, p={p_value:.4f} "
        f"({'significant at p<0.05' if p_value < 0.05 else 'not significant at p<0.05'})"
    )
    if b + c < 10:
        print(
            "Note: b + c < 10, so treat this p-value as indicative only — sample more questions "
            "or run with a higher --typo_rate for a more reliable test."
        )

    print(f"\nResults file (append-only, safe to resume): {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test whether typos in questions hurt LLM accuracy on MMLU"
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
    parser.add_argument(
        "--typo_rate",
        type=float,
        default=0.15,
        help="Fraction of eligible words in the question stem to perturb",
    )
    parser.add_argument("--seed", type=int, default=42)

    run(parser.parse_args())
