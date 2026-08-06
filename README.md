# Evaluation Lab
A public research notebook on LLM evaluation, benchmarking, and reliability.

This repository documents my journey studying how we evaluate large language models, RAG systems, and AI agents. It contains paper reviews, benchmark reproductions, experiments, and original evaluation ideas.

## Why?
LLMs are improving rapidly, but measuring their capabilities reliably remains an open research problem.

Instead of focusing on building more applications, this repository explores questions such as:

- How should we evaluate LLMs?
- What makes a good benchmark?
- When can LLMs judge other LLMs?
- How should RAG and agentic systems be evaluated?
- Which evaluation metrics correlate with human judgment?

## Repository Structure

```
utilities/              # Shared eval toolkit (LLM clients, MCQ parsing, checkpoints)
benchmarks/
    <BenchmarkName>/
        notes/          # Paper PDF, summary, and reading notes
        analysis/       # Dataset limitations and observations
        experiments/    # Python evaluation scripts
datasets/
    <BenchmarkName>/
        data/           # Raw CSV splits (dev, val, test)
experiments/            # Cross-benchmark experiment notebooks
```

### Shared utilities

`utilities/` holds reusable pieces for future papers (HELM, TruthfulQA, …):

| Module | Role |
|--------|------|
| `llm.py` | Multi-provider client: `ollama`, `groq`, `openai`, `anthropic`, `gemini` |
| `mcq.py` | A–D prompting + answer extraction |
| `checkpoint.py` | Append-only CSV resume helpers |
| `env.py` / `paths.py` / `cli.py` | `.env` loading, dataset paths, shared CLI flags |

Put API keys in a root `.env` (see `.gitignore`):

```
GROQ_API_KEY=...
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GEMINI_API_KEY=...   # or GOOGLE_API_KEY
```

Example:

```bash
python benchmarks/MMLU/experiments/evaluate_mmlu.py \
  --provider openai --model gpt-4o-mini --subject anatomy --limit 10 --json
```

## Current Learning Roadmap
- [x] MMLU
- [ ] HELM
- [ ] TruthfulQA
- [ ] SWE-Bench
- [ ] Arena-Hard
- [ ] AlpacaEval
- [ ] BrowserArena
- [ ] GAIA
- [ ] DeepEval
- [ ] LM Evaluation Harness

## Current Interests

- LLM evaluation
- Benchmark design
- Agent evaluation
- RAG evaluation
- LLM-as-a-Judge
- Evaluation reliability
- Statistical analysis of benchmarks

| Paper | Status | Notes |
|--------|--------|-------|
| MMLU | ✅ | Completed analysis, bias detection, and local/API evaluation |
| HELM | ⏳ | Planned |
| TruthfulQA | ⏳ | Planned |
| SWE-Bench | ⏳ | Planned |

## Open Questions

- Can retrieval metrics predict hallucination?
- Can agent trajectories be evaluated before task completion?
- When do LLM judges disagree with humans?
- How stable are benchmark rankings over time?
- Can benchmark contamination be detected automatically?

## Goals

- Build an open-source evaluation toolkit
- Publish an evaluation benchmark
- Contribute to LM Evaluation Harness
- Write technical blogs on evaluation
- Conduct reproducible benchmark studies

## Principles

This repository is not a collection of paper summaries.

For every paper I study, I aim to answer:

- What problem does this evaluation solve?
- What assumptions does it make?
- What are its limitations?
- Can I reproduce the results?
- Can I improve the methodology?