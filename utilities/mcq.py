"""
mcq.py

Multiple-choice (A/B/C/D) prompting and answer parsing for MMLU-style evals.

Functions
---------
  format_question(question, options, include_answer=False, correct_label=None)
      Build a standard "Question: … / A. … / Answer:" prompt.
  generate_few_shot_prefix(dev_df, num_shots=5)
      Concatenate labeled examples from a labeled MCQ DataFrame.
  extract_answer(model_output)
      Parse A/B/C/D from free text or {"answer": "X"} JSON; else "N/A".

No CLI arguments — library helper only.
"""

from __future__ import annotations

import json
import re
from typing import List, Optional

import pandas as pd


def format_question(
    question: str,
    options: List[str],
    include_answer: bool = False,
    correct_label: Optional[str] = None,
) -> str:
    """Standard A–D multiple-choice prompt used by MMLU-style evals."""
    prompt = f"Question: {question}\n"
    prompt += f"A. {options[0]}\n"
    prompt += f"B. {options[1]}\n"
    prompt += f"C. {options[2]}\n"
    prompt += f"D. {options[3]}\n"
    prompt += "Answer:"
    if include_answer and correct_label:
        prompt += f" {correct_label}\n\n"
    return prompt


def generate_few_shot_prefix(dev_df: pd.DataFrame, num_shots: int = 5) -> str:
    """Build a few-shot prefix from the first num_shots rows of a labeled MCQ frame."""
    prefix = ""
    shots = dev_df.head(num_shots)
    for _, row in shots.iterrows():
        options = [row["A"], row["B"], row["C"], row["D"]]
        prefix += format_question(
            row["question"],
            options,
            include_answer=True,
            correct_label=str(row["label"]).strip(),
        )
    return prefix


def extract_answer(model_output: str) -> str:
    """
    Pull an A/B/C/D choice from free-text or JSON model output.

    Preference order:
      1. JSON object with an "answer" key
      2. Bare letter (optionally wrapped in punctuation)
      3. Last standalone A–D token in the text
    """
    cleaned = model_output.strip()
    if not cleaned:
        return "N/A"

    try:
        data = json.loads(cleaned)
        if isinstance(data, dict) and "answer" in data:
            ans = str(data["answer"]).strip().upper()
            if ans in ["A", "B", "C", "D"]:
                return ans
    except Exception:
        pass

    short_match = re.match(r"^\s*\(?([A-Da-d])\)?\.?\s*$", cleaned)
    if short_match:
        return short_match.group(1).upper()

    matches = re.findall(r"\b([A-D])\b", cleaned)
    if matches:
        return matches[-1]

    return "N/A"
