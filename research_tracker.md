# Active Research Questions

## RQ-1
Do LLMs exhibit systematic answer-position bias in multiple-choice benchmarks?

**Status:** Completed (MMLU)

**Finding:** Both models were sensitive to answer-option ordering, indicating that predictions are not fully invariant to answer position.

---

## RQ-2
Do LLMs disproportionately prefer longer answer options?

**Status:** Completed (MMLU)

**Finding:** Yes. Both models selected the longest or second-longest option substantially more often than expected by chance, with Llama 3.2 exhibiting a stronger preference.

---

## RQ-3
Does answer-option ordering affect model performance?

**Status:** Completed (MMLU)

**Finding:** Yes. Shuffling answer options changed accuracy for both models, with GPT-5 Nano showing larger overall changes and strong subject-specific effects.

---

## RQ-4
Do answer choices themselves contain predictive information independent of the question?

**Status:** Completed (MMLU)

**Finding:** Yes. Both models achieved well above the 25% random baseline using only answer options, suggesting that answer choices contain exploitable information.

---

## RQ-5
Does lexical negation systematically affect model performance?

**Status:** Completed (MMLU)

**Finding:** No significant overall effect was observed. Negation sensitivity appears to depend on the subject rather than acting as a universal source of difficulty.

---

## RQ-6
How robust are LLMs to light typographical noise?

**Status:** Completed (MMLU)

**Finding:** Both models experienced small but statistically significant accuracy drops (~2 percentage points), indicating measurable but limited sensitivity to light spelling errors.

---

## RQ-7
Does in-context learning (few-shot) consistently improve performance?

**Status:** Planned

---

## RQ-8
How sensitive are models to distractor quality?

**Status:** Planned

---

## RQ-9
How does removing supporting context affect reasoning?

**Status:** Planned

---

## RQ-10
Does model confidence correlate with robustness?

**Status:** Planned