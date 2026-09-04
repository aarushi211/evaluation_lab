## Motivation
- Models have begun to plateau, making it difficult to discern their abilities. 
- Extend MMLU: more challenging, reasoning focused questions, expanding choices from 4 to 10.
- Eliminates trivial and noisy questions in MMLU.
- Drop in accuracy by 16 to 33%
- Stability under diff prompts.
- 24 diff prompts tested. 4-5% in MMLU but only 2$ in MMLU-Pro.
- CoT performed better on MMLU-Pro unlike with MMLU. 

## Problems with MMLU
1. Only 3 distractors. LLMs can exploit shortcuts to derive the ans leading to overestimation of LLMs performance and leading to a degree of instability.
2. MMLU is knowledge driven and does not require reasoning and achieve better results when answered directly.
3. There is portion of questions that are either unanswerable or mistakenly annotated. This leads to lower ceiling which the frontier models hit.

## MMLU-Pro
- MMLU-Pro spans 14 diverse domains - maths, physics, law, engineering etc with over 12k Qs.
- Has 10 options. reducing the probability to guess and increases the difficulty and robustness.
- Increase in portion of college level exam problems. Require LLMs to deliberate reasoning. 
- 2 rounds of expert reviews to reduce noise. First is expert verification and second uses SoTA LLMs to identify potential errors and employ annotators to perform more targeted verification. 

## Key Findings
- GPT-4o achieves 72.6% and GPT-4-Turbo achieves 63.7% accuracy.
- More discriminative. Gap b/w GPT-4o and GPT-4-Turbo in MMLU was just 1% while in MMLU Pro is 9%.
- Open source models thoug not at a level of closed source still showed performace close to Clude-3-Sonnet.
- MMLU-pro requires CoT to achieve promising results. Boosted GPT-4o by 19%.
- Error analysis of GPT-4o revealed -
    - 39% due to reasoning process
    - 35% lack of domain knowledge
    - 12% from computational errors. 

## Dataset 
1. Original MMLU
2. STEM Website
3. TheoremQA
- Scibench

### Dataset Construction Pipeline
- For MMLU dataset, 57 categories reduced to 14 and 8 models were evaluated. Q answered correctly by more than 4 models considered easy and removed. 
> Did the design changes introduced by MMLU-Pro actually reduce the benchmark sensitivities observed in MMLU?
- STEM website had discriptive QA, so GPT-4-Turbo was used to extract short answer and additional distractors for each Q. Incomplete and incorrect removed manually.
- GPT-4-Turbo used to add additional distractor options. Also experimented to ensure that GPT-4-Turbo does not gain additional advantage from such augmentation procedure. 
> Do GPT-generated distractors introduce new structural artifacts?
> Can a model distinguish the original answer/options from generated distractors?
> Does answer-only accuracy remain above chance when chance is ~10% rather than 25%?
- Expert review. Phase 1 was verification of correctnedd and appropriateness which included verifying accuracy, removing unsuitable questions, etc. Phase 2 was ensuring distractor validity which involved using Gemini-5-pro to re-evaluate options to identify false negatived. Then experts manually reviewed them. 

## Experimental Setup
- Utilized 5 shot CoT approach. 
    - Extending original options available from CoT Though Hub
    - Selected 5 demostration examples for each discipline. 
- For ans extraction 2 regex are used. If both fail the fallback mechanism selected an option at random 
> How frequently does answer extraction fail and trigger the random fallback? Since extraction failures are replaced with a random answer rather than counted as incorrect, could differing extraction-failure rates introduce noise or bias into model comparisons?

## Error Analysis of GPT-4o
- Reasoning Errors (39%): Difficulties with logical reasoning even when it recalls correct info and knowledge. Likely due to dependence on recognizing patterns in training data
- Lack of Specified Knowledge (35%): Errors such as incorrect financial calculations and misapplications of optical principles highlight this issue.
- Calculation Errors (12%): Found instances where model had correct formula but makde computation mistakes.
- Other errors: No selection error, question understanding error, generation ussues, annotation errors, ans extraction errors. Attributed to limitation in final response selection, complex text interpretation challenges, limitation in scope, etc.

## Limitations
- Limitations of MCQ format. Cannot capture the depth of comprehension & creative generations
- Does not assess multi-model models.