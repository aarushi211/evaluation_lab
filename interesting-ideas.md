# Interesting Ideas

## Dynamic MMLU

Generate distractors using another LLM instead of fixed answer options.

---

## Confidence Calibration

Instead of only predicting A/B/C/D, ask the model for:

- answer
- confidence
- explanation

Compare calibration under perturbations.

---

## MMLU-Noise

Create multiple noisy versions of every MMLU question:

- typos
- OCR errors
- missing punctuation
- grammar mistakes

Measure robustness.

---

## Option Generation

Replace distractors with GPT-generated distractors.

Does answer-only accuracy disappear?

---

## Cross-Benchmark Robustness

Do the perturbations discovered in MMLU also affect

- GPQA
- TruthfulQA
- MMLU-Pro
- HellaSwag

---

## Quantization Robustness

Do GGUF models become more sensitive to perturbations than FP16 models?

---

## Retrieval vs Closed Book

Run

Original model

↓

RAG-enabled model

↓

Same perturbations

Compare robustness.

# MMLU-Pro
- Does answer-only performance remain above chance with ~10 options?
- Did increasing the number and quality of distractors actually reduce
  answer-choice artifacts?

- Does option shuffling still substantially change performance?
  - MMLU-Pro claims greater prompt robustness, but that does not
    necessarily imply option-order robustness.

- Are correct-answer positions balanced across A-J?
- Is answer length correlated with correctness?
- Do GPT-4-Turbo-generated distractors have detectable structural
  differences from correct answers/original options?

- How often does answer extraction fail?
  - Does the random fallback affect reported accuracy?
  - Does extraction failure vary by model?

- Does the 56.6% MMLU-derived subset behave differently from questions
  originating from STEM Website / TheoremQA / SciBench?

- Are there duplicates or cross-source overlaps?

- Does typo robustness improve compared with MMLU?
- Does lexical negation show similar subject/domain-level behavior?