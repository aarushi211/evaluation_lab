# MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark
### Assumptions
- Accuracy is sufficient as the primary metric for comparing model performance.
- Multiple-choice questions can be used to measure language understanding and reasoning.
- Increasing the number of distractors reduces the possibility of using shortcuts compared with MMLU.
- Increasing the number of options from 4 to up to 10 makes random guessing substantially harder.
- Adding more challenging, reasoning-heavy questions makes the benchmark more discriminative as models improve.
- College-level and reasoning-focused questions require more deliberate reasoning and are less likely to be solved through simple memorization or pattern matching.
- Expert review and filtering improve the quality of the benchmark.
- Greater robustness to prompt variations indicates a more stable benchmark.

### Possible Weaknesses
- Multiple-choice questions still allow guessing.
- Accuracy is still the primary evaluation metric.
- The benchmark is English-only.
- It does not test multimodal understanding.
- It does not evaluate long-form/open-ended generation.
- Increasing the number of distractors may reduce some shortcuts while introducing new ones.
- Many additional distractors were generated using GPT-4-Turbo, which could introduce model-generated patterns or artifacts into the benchmark.
- Questions come from multiple sources and undergo different filtering/construction procedures, which could introduce source-specific artifacts.
- Some questions have fewer than 10 options, so the random baseline is not identical across all questions.
- The answer-extraction procedure falls back to a random option when both extraction methods fail. The effect depends on how frequently this actually occurs.
- Benchmark contamination is still possible, especially for the portion inherited from the original MMLU dataset.

### Questions
- Did the design changes introduced by MMLU-Pro actually reduce the benchmark sensitivities observed in MMLU?
- Does answer-only performance remain above chance when there are approximately 10 options instead of 4?
- Does shuffling answer choices still affect performance?
- Are correct-answer positions uniformly distributed across A-J?
- Is answer length still correlated with correctness?
- Do GPT-generated distractors introduce new structural artifacts?
- Can models distinguish GPT-generated distractors from the original answer choices?
- Do results differ based on question source: MMLU, STEM Website, TheoremQA, and SciBench?
- Does the MMLU-derived subset behave differently from the newly introduced questions?
- Are there duplicates or overlaps within the dataset or between its source datasets?
- How frequently does answer extraction fail and trigger the random fallback?
- If extraction-failure rates differ across models, could the random fallback affect model comparisons?
- Why use random selection after extraction failure instead of marking the response incorrect?
- Does robustness to prompt variations also imply robustness to other semantically irrelevant perturbations such as answer ordering or small typos?
- Does Chain-of-Thought improve actual reasoning, or does the benchmark simply reward models that perform better when given additional inference tokens?
- Are the improvements from CoT consistent across domains?
- Does increasing the number of options improve benchmark quality, or primarily make the task harder?

### My Takeaways
- MMLU-Pro addresses several known weaknesses of MMLU rather than simply making the questions harder.
- Benchmark design has to evolve as model performance approaches saturation.
- Increasing the number of answer choices can make guessing and shortcut-based strategies more difficult, but it does not guarantee that structural artifacts disappear.
- A benchmark can be more difficult without necessarily being more robust.
- Prompt robustness is one type of robustness. It does not establish robustness to answer ordering, formatting, textual noise, or other perturbations.
- Dataset construction choices, including filtering, source selection, and generated distractors, become part of what the benchmark measures.
- MMLU-Pro provides an interesting comparison with MMLU because several of its design decisions explicitly attempt to address limitations of the original benchmark.

### Research Questions
- Do the structural signals I observed in MMLU persist in MMLU-Pro?
- Does increasing from 4 to approximately 10 answer choices reduce answer-only predictive performance?
- Does MMLU-Pro reduce model sensitivity to answer-choice ordering?
- Are GPT-generated distractors systematically distinguishable from human-written/original choices?
- How much do answer length and answer position predict correctness in MMLU-Pro?
- Do structural artifacts differ depending on the original source of the question?
- Are MMLU-derived questions more susceptible to answer-only prediction or option-order sensitivity than newly sourced questions?
- How robust is MMLU-Pro to controlled perturbations such as option shuffling and textual noise?
- Can benchmark contamination be separated from structural answer-choice signals?