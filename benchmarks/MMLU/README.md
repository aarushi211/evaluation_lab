# MMLU Benchmark Investigation

This case study examines how structural properties and simple perturbations of [MMLU](https://arxiv.org/abs/2009.03300) affect model performance. The goal is not a leaderboard score, but to ask how much of that score is capability versus an artifact of the benchmark.

**Models**

- Llama 3.2 3B (local via Ollama)
- GPT-5 Nano (OpenAI API)

Unless noted otherwise, comparisons use the full MMLU **test** set (57 subjects, 14,042 questions).

---

## How to explore this folder

```
benchmarks/MMLU/
  README.md                 ← you are here
  notes/                    paper notes and summaries
  analysis/
    dataset_observation.md  dataset forensics (no models)
    experimental_findings.md model experiments + interpretation
  experiments/              runnable Python scripts
  results/                  CSV outputs from those scripts
```

| If you want… | Start here |
|--------------|------------|
| One-page findings | this README |
| Dataset quirks (length bias, leakage, duplicates) | [`analysis/dataset_observation.md`](./analysis/dataset_observation.md) |
| Full experiment write-up | [`analysis/experimental_findings.md`](./analysis/experimental_findings.md) |
| Paper reading notes | [`notes/`](./notes/) |
| How to rerun a probe | [`experiments/`](./experiments/) + commands below |
| Raw numbers | [`results/`](./results/) |

Shared LLM / checkpoint helpers live in the repo root [`utilities/`](../../utilities/).

---

## Key Findings

### 1. Answer-only evaluation

Removing the question reduced accuracy substantially, but both models remained well above the 25% random baseline (one-sample proportion *z*-tests, *p* ≪ 0.001). ~70% of picks were the longest or second-longest option.

| Model | Standard | Answer only |
|---|---:|---:|
| Llama 3.2 3B | 61.88% | 36.16% |
| GPT-5 Nano | 69.75% | 39.70% |

### 2. Option shuffling

Overall accuracy dropped modestly, but **subject-level swings were large** — especially for GPT-5 Nano (e.g. high-school math 44.8% → 94.8%; professional medicine 85.3% → 55.2%). Aggregate scores hide order sensitivity.

| Model | Standard | Shuffled options | Δ |
|---|---:|---:|---:|
| Llama 3.2 3B | 61.88% | 59.62% | −2.26 |
| GPT-5 Nano | 69.75% | 65.32% | −4.43 |

### 3. Dataset structure

From [`analysis/dataset_observation.md`](./analysis/dataset_observation.md):

- Correct answer is the longest option ~**35.9%** of the time (chance = 25%)
- Length bias and negation density vary sharply by subject / category
- Correct-answer positions are not uniform (D slightly over-represented)
- Cross-split overlap: **47** test↔val, **5** test↔dev
- Exact item duplicates (`question` + options): **27** within-subject + **78** cross-subject excess rows on test

### 4. Lexical negation

Within-subject matched samples (negation vs non-negation). **No significant overall effect** for either model; subject-level swings are large.

| Model | Negation | Non-negation | Two-proportion *z* |
|---|---:|---:|---|
| Llama 3.2 3B | 63.72% | 61.90% | *z* = 0.74, *p* = 0.46 |
| GPT-5 Nano | 73.23% | 76.80% | *z* = −1.62, *p* = 0.10 |

### 5. Small typos

Light stem typos produce a ~2 pt drop for both models. McNemar’s test on paired outcomes is highly significant — higher baseline accuracy did not mean greater robustness.

| Model | Original | Typo’d | McNemar |
|---|---:|---:|---|
| Llama 3.2 3B | 61.93% | 59.70% | χ² = 77.1, *p* < 0.0001 |
| GPT-5 Nano | 69.76% | 67.60% | χ² = 53.8, *p* < 0.0001 |

---

## Important caveat

Above-chance answer-only performance does **not** establish *why* the models achieve it. Structural signals in the options, semantic relationships among choices, prior exposure to MMLU-like material, or some combination could contribute. These experiments were **not** designed to prove benchmark contamination.

Similarly, shuffling and typos show *sensitivity*, not a full causal account of every subject-level swing.

---

## Reproduce the experiments

From the repo root (API keys in `.env` as documented in the main README):

```bash
# Standard / shuffled evaluation (full test set)
python benchmarks/MMLU/experiments/evaluate_mmlu.py \
  --provider openai --model gpt-5-nano --subjects all --limit 0 --json --workers 8

python benchmarks/MMLU/experiments/evaluate_mmlu.py \
  --provider openai --model gpt-5-nano --subjects all --limit 0 --shuffle --json --workers 8

# Answer-only baseline
python benchmarks/MMLU/experiments/answer_only_baseline_eval.py \
  --provider openai --model gpt-5-nano --subjects all --limit 100000 --json --workers 8

# Negation (within-subject matched groups)
python benchmarks/MMLU/experiments/negation_impact.py \
  --provider openai --model gpt-5-nano --subjects all --limit_per_group 20 --json --workers 8

# Typo robustness
python benchmarks/MMLU/experiments/typo_robustness.py \
  --provider openai --model gpt-5-nano --subjects all --limit 100000 --json --workers 8

# Dataset forensics (no model calls)
python benchmarks/MMLU/experiments/analyze_dataset.py
python benchmarks/MMLU/experiments/find_duplicate_questions.py --match all
```

Swap `--provider ollama --model llama3.2` for the local model. Use `--limit 10` for a smoke test.

| Script | Role |
|--------|------|
| [`evaluate_mmlu.py`](./experiments/evaluate_mmlu.py) | Standard / shuffled MCQ eval |
| [`answer_only_baseline_eval.py`](./experiments/answer_only_baseline_eval.py) | Options with question omitted |
| [`negation_impact.py`](./experiments/negation_impact.py) | Negation vs non-negation |
| [`typo_robustness.py`](./experiments/typo_robustness.py) | Paired clean vs typo’d stems |
| [`analyze_dataset.py`](./experiments/analyze_dataset.py) | Aggregate dataset stats |
| [`find_duplicate_questions.py`](./experiments/find_duplicate_questions.py) | Stem / item duplicates |
| [`find_specific_anomalies.py`](./experiments/find_specific_anomalies.py) | Missing values, identical options, leakage |
| [`subject_bias_analysis.py`](./experiments/subject_bias_analysis.py) | Category / subject bias tables |

---

## Results

Representative CSVs under [`results/`](./results/):

| Artifact | File pattern |
|----------|----------------|
| Standard eval | `*_all_no_shuffle_0shot_lim100000.csv` |
| Shuffled options | `*_all_shuffled_0shot_lim100000.csv` |
| Answer-only | `answer_only_*_lim100000.csv` |
| Negation | `negation_*_lim20.csv` |
| Typos | `typos_*_lim100000.csv` |
| Dataset summaries | `mmlu_*_summary.csv`, `mmlu_duplicate_questions_*.csv` |

---

## Full write-up

For discussion, figures, interpretation, and limitations:

- Dataset: [`analysis/dataset_observation.md`](./analysis/dataset_observation.md)
- Experiments: [`analysis/experimental_findings.md`](./analysis/experimental_findings.md)
- Substack Article: **[Click Here](https://aarushijain750597.substack.com/p/how-much-of-an-mmlu-score-comes-from?r=90nqwv&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)**
