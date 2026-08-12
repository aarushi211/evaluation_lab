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