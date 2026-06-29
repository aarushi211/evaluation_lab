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

papers/
    Paper analyses and critiques
experiments/
    Benchmark reproductions and evaluation studies
benchmarks/
    Original benchmark designs
datasets/
    Evaluation datasets
notes/
    Research questions and literature summaries

## Current Learning Roadmap

- [] HELM
- [] MMLU
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
| MMLU | 🚧 | Reading |
| HELM | ⏳ | Reading |
| TruthfulQA | ⏳ | Reading |
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