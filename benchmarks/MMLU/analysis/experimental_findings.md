# Experimental Findings

## Scope

This document reports model-based experiments designed to test whether structural characteristics identified during MMLU dataset analysis correspond to measurable differences in model behavior.

Unless otherwise specified, experiments compare:

- Llama 3.2 3B
- GPT-5 Nano

The goal is not to establish state-of-the-art MMLU performance, but to study the robustness and reliability of MMLU-style evaluation.

## Experiment 1 - Standard MMLU Evaluation
Measuring the baseline performance of the models before applying perturbations.

### Results
Model | Accuracy | Correct | Total Questions
------|----------|---------|----------------
Llama 3.2| 61.88 | 8689 | 14042|
GPT-5 nano| 69.75% | 9794 | 14042 |

### Observations
- Llama 3.2 has an overall accuracy of ~62% with highest accuracy of 82% in marketing and lowest of 30% in college mathematics.
- GPT-5 nano has an overall accuracy of ~70% with highest accuracy of ~94% in high school government and politics and lowest accuracy of 37% in college mathematics.

### Interpretation
- GPT-5 Nano performed better on almost every MMLU subject, meaning its advantage is not limited to a particular academic subject. 
- Both models appear to struggle with formal and mathematical reasoning suggesting that more structured and multi-step reasoning remains challenging. 
- GPT-5 Nano performs particularly well on several conceptual and knowledge-intensive subjects, suggesting that it is effective at retrieving and applying learned academic knowledge.
- The relatively low scores in professional law and accounting suggest that specialized knowledge and precise domain-specific reasoning remain difficult.

## Experiment 2 - Option Shuffling
Measures whether a model is invariant to answer-option ordering.

### Results
Model | Accuracy | 
------|----------|
Llama 3.2| 59.62%  | 
GPT-5 nano| 65.32% | 

### Observations
- Both models experienced a decrease in accuracy after answer options were shuffled. 
- GPT-5 nano experienced a larger performance drop from 69.75% -> 65.32%, which is a decrease of 4.43 percentage points. Llama on the other hand experienced a drop from 61.88% -> 59.62%, that is drop of 2.26 percentage points.
- The effect of shuffling varies considerably by subject, meaning some subjects experienced large decrease, while others show substantial improvements.
- GPT-5 nano shows extremely large improvements in several mathematical subjects.
    - High-school mathematics: 44.81% → 94.81%
    - Elementary mathematics: 54.23% → 90.74%
    - College mathematics: 37.00% → 90.00%
    - College physics: 53.92% → 87.25%
- Substantial decrease has been observed in several other subjects by GPT-5 nano too. 
    - Professional medicine: 85.29% → 55.15%
    - Security studies: 76.73% → 53.88%
    - Global facts: 43.00% → 37.00%
    - Professional law: 51.04% → 39.31%
- Llama 3.2 also shows some substantial subject-level variati
    - College mathematics: 30.00% → 28.00%
    - Abstract algebra: 37.00% → 29.00%
    - College physics: 37.25% → 32.35%
- The subjects that are strongest or weakest are not necessarily preserved after shuffling. For GPT-5 nano, its original strongest subject was high-school government and politics (93.78%), whereas after shuffling high-school mathematics became its strongest (94.81%).

### Interpretation
- The results suggest that neither model is completely invariant to answer-option ordering. If the models were fully invariant, randomly rearranging the options should have little or no effect on accuracy.
- GPT-5-nano appears more sensitive to answer-option ordering than Llama 3.2. Its accuracy decreased by 4.43 percentage points, compared with a 2.26-point decrease for Llama 3.2.
- However, the subject-level results show that option shuffling does not simply make questions harder. Some subjects become dramatically easier after shuffling, particularly mathematical subjects for GPT-5-nano.
- The large improvements in mathematics suggest that the original answer ordering may itself contain positional or structural information that affects model predictions. In other words, the model may be responding partly to the arrangement of choices rather than solely to the semantic content of the question.
- The results therefore provide evidence of possible answer-order sensitivity. The model's choice may depend, at least to some extent, on where an option appears in the prompt.
- The effect appears to be model-dependent. GPT-5-nano shows much larger changes in several subjects, particularly mathematics, whereas Llama 3.2 generally shows smaller changes in those same subjects.
- The experiment also demonstrates that aggregate accuracy alone can hide substantial behavioral changes. Although Llama's overall accuracy decreases by only 2.26 points, individual subjects can experience much larger changes.

## Experiment 3 - Answer-only Baseline
This experiments checks whether models are able to achieve above-chance accuracy when given answer options without questions. 

### Results
Model | Llama 3.2 | GPT-5 nano
------|--------|--------|
Accuracy | 36.16% | 39.70% |
Ground Truth is longest option | 35.88% | 35.93% |
Predicted longest option| 44.20% | 39.52% |
Predicted longest or 2nd longest option | 71.22% | 69.76% |
Mean predicted option length | 44.8 | 44.9 |
Z-test | z = 30.549, p < 0.0001 | z = 40.215, p < 0.0001 |

### Observations
- Both Llama 3.2 and GPT-5 nano have an accuracy above the random chance baseline of 25%.
- Both models predicted the longest option more frequently than it was correct.
- Both models favor the longest or the 2nd longest options.
- Both models average predicted length is similar.
- Several social science and knowledge heavy subjects remain relatively strong without questions.
- Mathematical, physics and moral scenarios subjects are generally close to chance.
- Llama 3.2 exhibits a stronger preference for longer answer options than GPT-5-nano in the options-only experiment.

### Interpretation
- The answer options themselves contain measurable information as giving only the options should theoretically leave it close to the 25% random baseline.
- The z-test also reinforces that the options-only accuracy is extremely unlikely to be explained by random guessing alone.
- The Llama vs GPT gap when questions were provided is 7.87 pts, while without questions, it is 3.54 pts. This suggests question context is important.
- The high longest or 2nd longest statistic reveals that both models have a systematic option-selection bias as if the 4 options were selected uniformly at random, roughly 50% of predictions should have fall within the 2 longest options.
- This experiment reveals that a non-trivial amount of predictive performance can be obtained without access to the question itself. This suggests that MMLU accuracy should not necessarily be interpreted as a pure measurement of question-answering ability.
- Social sciences and knowledge heavy subjects provide more information within the options when compared to Mmathematics, physics and moral science subjects. 

## Experiment 4 - Lexical Negation Impact
This experiment checks whether questions containing negative construction associate with lower model accuracy. 

### Results
Model | Negation Accuracy | Non-negation Accuracy | Tow-Propotion z-test|
------|-------------------|-----------------------|----------------------|
Llama 3.2 | 63.72% | 61.90% | z=0.740, p=0.4590 
GPT-5 nano | 73.23% | 76.80% | z=-1.624, p=0.1043

### Observations
- GPT-5 nano performs slightly worse on negation questions: accuracy decreases from 76.80% on non-negation questions to 73.23% on negation questions, a difference of −3.57 percentage points.
- Llama 3.2 shows the opposite pattern: accuracy is 63.72% on negation questions compared with 61.90% on non-negation questions, an increase of +1.82 percentage points.
- Neither GPT-5 Nano nor Llama 3.2 exhibits a statistically significant overall difference between negation and non-negation questions. Therefore, this experiment does not provide sufficient evidence that lexical negation alone systematically affects MMLU performance.
- The effect of negation varies considerably across subjects. For example, GPT-5 nano drops from 95% to 61.11% in computer security, while increasing from 25% to 52.63% in high-school mathematics. Llama 3.2 similarly shows large subject-level variations.
- "NOT" is by far the most common negation cue, with 603 questions, compared with 73 containing "EXCEPT". This means the overall negation result is dominated by questions containing NOT.
- For both models, EXCEPT questions appear relatively strong: GPT-5 nano achieves 80.82% and Llama 3.2 achieves 78.08%. However, the sample is much smaller than for NOT, so this should be interpreted cautiously.

### Interpretation
- The experiment does not support the hypothesis that negation universally hurts LLM performance. Neither model shows a statistically significant overall difference between negation and non-negation questions. Hence, the absence of a significant overall effect suggests that lexical negation is not, by itself, a dominant source of difficulty for these models on MMLU.
- The results suggest that negation sensitivity is likely context-dependent rather than a general model weakness. The impact appears to depend more strongly on the subject and potentially on the type of negation used.
- These results suggest an interaction between subject domain and negation. Negation may become difficult only when combined with domain-specific reasoning requirements, rather than acting as a standalone source of difficulty.
- Questions containing the cue "EXCEPT" exhibit relatively high accuracy for both models. However, because only 73 such questions were available, this observation should be considered exploratory rather than conclusive.

## Experiment 5 - Typo Robustness
This experiment introduces small typos in the questions that mimics a human typing mistake, and helps evaluate the model robustness.

### Results
Model | Original Accuracy | Typo'd Accuracy | Flipped correct -> incoorect due to typo | Flipped incorrect -> correct due to typo | McNemar's test
------|----------|---------|----------------| --------------|----------|
Llama 3.2| 61.93% | 59.70% | 779| 468 | chi2=77.065, p=0.0001
GPT-5 nano| 69.76% | 67.60% | 993 | 691 | chi2=53.801, p=0.0001

### Observations
- Both models experienced statistically significant drops of approximately two percentage points.
- More questions changed from correct to incorrect than vice-versa.
- Both McNemar's tests are highly significant.
- Total flips by GPT-5 nano is 993 + 691 = 1,684 is more than llama 3.2 which has 779 + 468 = 1,247

### Interpretation
- Both models are sensitive to minor textual corruptions, though measurable, its relatively small.
- Despite GPT-5 Nano's substantially higher baseline accuracy, both models experience almost identical absolute degradation under light typographical noise.
- The McNemar's test with highly significant p-values provide a strong evidence that the perturbation has a systematic effect.
- GPT-5 nano has more prediction instability, that is GPT-5 nano has more changed predictions than Llama 3.2, though it canceled out because of more incoorect -> correct flips. This suggest that GPT-5-nano may be more responsive to the altered input, but its additional changes are not exclusively harmful.

# Summary
Across the five experiments, MMLU performance was influenced not only by model capability but also by structural properties of the benchmark itself. Answer-option ordering, answer-option content, lexical negation, and minor typographical changes all affected model behavior, although the magnitude and direction of these effects varied across models and subject domains.

### Overall Findings
- GPT-5 Nano outperformed Llama 3.2 at baseline, achieving 69.75% compared with 61.88%, but higher accuracy did not consistently mean greater robustness.
- Answer-option ordering affected both models. Shuffling options reduced overall accuracy, particularly for GPT-5 Nano, showing that model predictions can depend on option position.
- Answer choices themselves contain exploitable information. Both models performed substantially above the 25% random baseline when given only the answer options, indicating that MMLU options contain predictive signals independent of the question.
- Negation did not significantly affect overall performance. Neither model showed a statistically significant difference between negation and non-negation questions, although effects varied across subjects.
- Minor typos significantly affected both models. Both experienced small accuracy decreases, with McNemar's tests confirming that the changes were statistically significant.
- Robustness varies by model and perturbation. GPT-5 Nano generally achieved higher accuracy but showed larger changes under some perturbations, demonstrating that higher benchmark performance does not necessarily imply greater robustness.
- Subject-level variation is substantial. The impact of perturbations differed considerably across MMLU subjects, suggesting that domain difficulty interacts with structural and textual characteristics.
- Interestingly, structural perturbations (answer ordering and answer-only evaluation) produced larger behavioral changes than the linguistic perturbations investigated here (negation and light typographical noise).

### Overall Conclusion
Taken together, these experiments suggest that MMLU accuracy reflects not only a model's ability to answer questions, but also its interaction with structural properties of the benchmark itself. While MMLU remains a valuable measure of broad knowledge, aggregate accuracy alone cannot fully characterize model robustness or reasoning ability. Evaluating models under controlled perturbations—such as answer-order changes, answer-only baselines, and textual noise—provides complementary insights that are hidden by leaderboard scores alone.