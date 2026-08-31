# Evaluation Lab

**Stress-testing how we measure LLMs - not just reporting another leaderboard score.**

Benchmarks like MMLU, HELM, and TruthfulQA are treated as ground truth for model quality. This repository asks a more basic question: **how much of a score is capability, and how much is an artifact of the benchmark?**

This is a **multi-benchmark research lab**. Each paper gets the same two-layer treatment; MMLU is only the first case study.

1. **Dataset forensics** - label skew, length bias, duplicates, split leakage, linguistic artifacts (negation, catch-alls, …).
2. **Controlled interventions** - hide the question, shuffle options, perturb the stem, swap distractors - then test whether those quirks actually move accuracy.

The goal is a reusable protocol and toolkit, not a one-off MMLU write-up.

---
## First Investigation: MMLU

I started with MMLU and tested how model performance changes when information that should be irrelevant to the underlying task is modified.

Some of the experiments include:
- removing the question entirely
- shuffling answer choices
- analyzing answer-length and position patterns
- testing lexical negation
- introducing small typos

📖 **Read the full investigation:** [Substack article](https://aarushijain750597.substack.com/p/how-much-of-an-mmlu-score-comes-from?r=90nqwv&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)

🔬 **Experiments, results, and analysis:** [`benchmarks/MMLU/`](benchmarks/MMLU/)

---

## Why this exists

Headline numbers are easy to cite and hard to interpret. If a heuristic (e.g. “pick the longest option”) already beats chance, if models score well with the *question omitted*, or if a small typo systematically flips answers, then the published score is not a pure measure of the skill the benchmark claims to test.

This lab is for:

- **Researchers / faculty** who care about evaluation validity, not only SOTA tables
- **Hiring managers** looking for experiment design, reproducible eval code, and statistical claims tied to artifacts

It is *not* an app demo, a wrapper around someone else’s harness, or a dump of paper summaries.

---

## Method (same for every benchmark)

```
ingest splits → forensic scripts → dataset note
      → intervention experiments (paired across models)
      → findings note + CSVs
```

**Eval engineering** (so runs stay comparable as the lab grows):

- Multi-provider client: Ollama, Groq, OpenAI, Anthropic, Gemini
- Append-only CSVs; resume after rate limits / crashes
- `question_id` = hash(content) for matching; `row_id` = hash(subject + row) so duplicates still run
- Shared CLI, seeds, and workers so two models see the same items
---

## Case studies

| Benchmark | Dataset forensics | Model interventions | Notes |
|-----------|-------------------|---------------------|--------|
| **MMLU** | Done | In progress | First full pass of the protocol. [Data](benchmarks/MMLU/analysis/dataset_observation.md) · [Experiments](benchmarks/MMLU/analysis/experimental_findings.md) |
| HELM | Planned | - | Next; same two-layer template |
| TruthfulQA | Planned | - | |
| SWE-Bench / agents | Later | - | Trajectories, not only MCQ |

---

## Repository layout

```
utilities/                 Shared toolkit (providers, MCQ parse, checkpoints, CLI)
benchmarks/
  <Name>/
    notes/                 Paper notes
    analysis/              Dataset observations + experimental findings
    experiments/           Stats + eval scripts
    results/               Run CSVs
datasets/<Name>/           Local splits (test / dev / val)
```

Adding a benchmark means a new folder that follows this template, plus whatever `utilities/` already provides.

---

## Toolkit (`utilities/`)

| Module | Role |
|--------|------|
| `llm.py` | `LLMEvaluator` — ollama / groq / openai / anthropic / gemini |
| `mcq.py` | Prompt formatting and A–D extraction |
| `checkpoint.py` | `question_id` / `row_id`, resume, append-only CSV |
| `cli.py` / `parallel.py` | Shared flags (`--provider`, `--data_dir`, `--workers`) |

API keys in a root `.env` (not committed): `OPENAI_API_KEY`, `GROQ_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`.

```bash
python benchmarks/MMLU/experiments/evaluate_mmlu.py \
  --provider openai --model gpt-4o-mini --subject anatomy --limit 10 --json
```

---

## How to read this repo

1. This README - scope of the lab.
2. The case-study table - which benchmark is in which stage.
3. That benchmark’s `analysis/` - claims and numbers.
4. `experiments/` + `utilities/` - how to rerun them.

If a number is not in those files, it does not belong in the notes.
