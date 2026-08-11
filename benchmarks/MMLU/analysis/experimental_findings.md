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
- GPT-5 nano appears to be particularly strong in conceptual and knowledge intensive subjects suggesting strong retrieval and application of learned academic knowledge.
- The relatively low scores in professional law and accounting suggest that specialized knowledge and precise domain-specific reasoning remain difficult.

## Experiment 2 - Option Shuffling
Measures whether a model is invariant to answer-option ordering.

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

## Experiment 4 - Negation Impact
This experiment checks whether questions containing negative construction associate with lower model accuracy. 

### Results


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
- Though GPT-5 nano starts at a higher baseline of ~70%, the absolute degradation is almost identical.So higher baseline capability does not automatically translate into greater robustness to input noise.
- The McNemar's test with highly significant p-values provide a strong evidence that the perturbation has a systematic effect.
- GPT-5 nano has more prediction instability, that is GPT-5 nano has more changed predictions than Llama 3.2, though it canceled out because of more incoorect -> correct flips. This suggest that GPT-5-nano may be more responsive to the altered input, but its additional changes are not exclusively harmful.