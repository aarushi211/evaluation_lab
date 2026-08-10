# Dataset Observations

## Introduction

This document summarizes **descriptive observations** obtained from programmatic analysis of the MMLU dataset. The goal is to characterize structural properties of the benchmark without making claims about model behavior or benchmark performance.

> **Scope:** These observations are based solely on the dataset contents. No model evaluation or heuristic experiments have been performed.

## Dataset Overview
The MMLU benchmark contains **57 subjects** organized into three evaluation splits.

Split       |   Questions |  Subjects
------------| ----------- | ----------
Dev|                  285  |       57|
Validation|         1,531 |        57|
Test|              14,042|         57|

- `dev`: Used for few shot prompting. Contains exactly 5 examples per subject. 
- `val`: Small validation set used for hyperparameter tuning or prompt validation to avoid overfitting to the test dataset.
- `test`: This is the primary evaluation set which the researchers have used when reporting the scores.
- `auxiliarr_train`: This contains broader, larger-scale training data grouped by overarching categories (like STEM, Humanities, Social Sciences) rather than the 57 granular subjects. It's used if someone wants to actually fine-tune a model on MMLU-like data.

**Legacy Text Files** (`devdata` and `fulldata`)
These folders typically represent raw or concatenated versions of the data, often formatted for specific early ingestion scripts used by the original paper's authors.
- `devdata` (5 txt): Often groups the development data by broader disciplines.
- `fulldata` (74 txt): Includes a mix of all subjects and auxiliary data combined into flat text formats for easy tokenization or quick text-parsing.

## Answer Position Distribution
Across the test split, the correct answer positions are:

Option   |  Percentage
--------| ------------
A       |      22.95%
B       |       24.65%
C       |       25.51%
D       |       26.89%

- The distribution is not perfectly uniform.
- Option **D** occurs slightly more frequently than other answers.
- Humanities exhibits a mild preference towards **C** (27.50%) while STEM (28.53%) and Social Science (30.60%) show a stronger preference towards **D**. 
Check [answer_position_bias.csv](../results/answer_position_bias_test.csv) for more detailed breakdown.

## Option Length Characteristics
The correct answer is the longest option in:

Split       |   Longest-option Rate
------------| ---------------------
Dev         |                38.25%
Validation  |                34.55%
Test        |                35.91%

**Subject Level Observations** <br>
- Subjects with strongest longest-option bias include:
    -   global_facts
    -   elementary_mathematics
    -   high_school_mathematics
    -   college_chemistry
    -   college_physics
- Subjects with the weakest longest-option bias include:
    -   public_relations
    -   econometrics
    -   human_aging
    -   conceptual_physics
    -   anatomy

- We can observe that option-length bias is considerable across subjects.
- Several STEM-related subjects exhibit substantially higher longest-option frequencies than the benchmark average.

## Catch-all Options
Catch all answers (e.g., *None of the above*, *All of the above*) appear 535 (3.81%) of test questions.

Academic Category  |     Catch-all Rate
------------------ |     ----------------
Humanities         |            36.13%
STEM               |             43.23%
Others (Applied/Professional)|   54.10%
Social Sciences|                 66.67%
Overall           |              44.86%

- Most catch-all options appear in **Option D**
- When present, the catch-all option is correct more frequently than random chance (25%).
- Though catch-all answers constitute only a small fraction of the overall benchmark, it still represents a recurring pattern.

## Negation Phrases
Negation phrases include **NOT**, **EXCEPT**, **FALSE**, **INCORRECT** etc.

Category                      |   Negation Frequency
------------------------------| --------------------
Humanities                    |               44.07%
Other (Applied/Professional)  |               21.01%
Social Sciences               |                7.49%
STEM                          |                7.13%

- Humanities contains substantially more negated questions than the remaining categories.
-   Negation frequency varies considerably between subjects, with several Humanities subjects acting as outliers.

## Cross-Split Coverage
An overlap between dataset questions has been observed.
Split Pair                |   Overlapping Questions
--------------------------| -----------------------
Test ↔ Validation         |                      47
Test ↔ Development        |                       5
Validation ↔ Development  |                       1

- Though cross-split overlap exists, it is concentrated within a small number of subjects, howvever, it is still important to note of such overlaps in questions between the splits.

## Duplicate Answers
A small number of duplicated options observed mostly in the test split.

Split | Count of Duplicate Options
------|---------------------
Test| 9
Dev | 1
Val | 3

- 5 questions duplicates the correct answer itself.
