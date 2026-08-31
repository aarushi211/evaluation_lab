# Dataset Observations

## Introduction

This document summarizes **descriptive observations** and their **potential limitations** identified from programmatic analysis of the MMLU dataset. The goal is to characterize structural properties of the benchmark without making claims about model behavior or benchmark performance.

> **Scope:** These observations are based solely on the dataset contents. No model evaluation or heuristic experiments have been performed, hence should not be interpreted as evidence that language model exploit these patterns or benchmark scores are inflated. 

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

### Key Observations:
- The distribution is not perfectly uniform.
- Option **D** occurs slightly more frequently than other answers.
- Humanities exhibits a mild preference towards **C** (27.50%) while STEM (28.53%) and Social Science (30.60%) show a stronger preference towards **D**. 
- Positional imbalance may introduce weak positional prior. 

## Option Length Characteristics
The correct answer in most human-designed multiple-choice questions is that they are often more detailed and nuanced, making it statistically longer than the incorrect distractors. 

The correct answer is the longest option in:

Split       |   Longest-option Rate
------------| ---------------------
Dev         |                38.25%
Validation  |                34.55%
Test        |                35.91%

### Subject Level Observations
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

### Key Observations:
- We can observe that option-length bias is considerable across subjects.
- Several STEM-related subjects exhibit substantially higher longest-option frequencies than the benchmark average.

### Potential Consequences
If a model simply employs a heiristic to **always predict the longest option**, it will achieve **~36% accuracy** on the test set, significantly outperforming the random chance bias (25%)

## Catch-all Options
Catch all answers (e.g., *None of the above*, *All of the above*) appear 535 (3.81%) of test questions.

Academic Category  |     Catch-all Rate
------------------ |     ----------------
Humanities         |            36.13%
STEM               |             43.23%
Others (Applied/Professional)|   54.10%
Social Sciences|                 66.67%
Overall           |              44.86%

### Key Observations:
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

### Key Observations:
- Humanities contains substantially more negated questions than the remaining categories.
-   Negation frequency varies considerably between subjects, with several Humanities subjects acting as outliers.

## Cross-Split Data Leakage
To ensure clean evaluations, there should be zero overlap between the different split pairs. However, some leakage across the splits have been observed.

Split Pair                |   Overlapping Questions
--------------------------| -----------------------
Test ↔ Validation         |                      47
Test ↔ Development        |                       5
Validation ↔ Development  |                       1

### Example Leakage Cases

**Contractile Proteins**:
   - **Test File**: `clinical_knowledge_test.csv` (Line 15, Label: B)
   - **Val File**: `college_medicine_val.csv` (Line 11, Label: B)
   - **Question**: *"The two principal contractile proteins found in skeletal muscle are:"*
2. **Endurance Sports**:
   - **Test File**: `clinical_knowledge_test.csv` (Line 73, Label: D)
   - **Val File**: `college_medicine_val.csv` (Line 7, Label: D)
   - **Question**: *"Which of the following physiological characteristics is not important for success in endurance events such as the marath..."*

### Key Observations:
- Though cross-split overlap exists, it is concentrated within a small number of subjects.
- However, repeated questions represent a potential source of evaluation contamination if validation examples are seen before testing.

## Duplicate Answers
Some questions offer duplicate choices, which reduces the effective number of options or introduces ambiguity.

5 questions duplicates the correct answer itself.

Split | Count of Duplicate Options
------|---------------------
Test| 9
Dev | 1
Val | 3


### Example Cases
| Subject File | Line No. | Question | Choices |
| :--- | :--- | :--- | :--- |
| elementary_mathematics_val.csv | 37 | *What is the value of \|3 + 5\| − \|−4\|?* | **A**: 12, **B**: -4, **C**: 4, **D**: 12 (A & D identical) |
| business_ethics_test.csv | 3 | *______ are the obligations of workers...* | **A**: Employee rights, **B**: Employee rights, **C**: Employer duties, **D**: Employee duties (A & B identical) |
| high_school_chemistry_test.csv | 141 | *When potassium perchlorate dissolves in water...* | **A**: ...spontaneous because it is exothermic, **D**: ...spontaneous because it is exothermic (A & D identical) |

## Duplicate Questions
In addition to identical choices within a single row, the same evaluation item can appear more than once, either inside one subject file or shared across two subjects in the same split.

*Same stem (question) — screening**

| Split | Scope | Duplicate groups | Duplicate rows | Excess rows | Max group size |
| :--- | :--- | ---: | ---: | ---: | ---: |
| Test | Within-subject | 78 | 163 | 85 | 6 |
| Test | Cross-subject | 80 | 180 | 100 | 10 |
| Dev | Within-subject | 1 | 2 | 1 | 2 |
| Val | Within-subject | 1 | 2 | 1 | 2 |
| Val | Cross-subject | 2 | 4 | 2 | 2 |

**Same item (question_options) — primary**

| Split | Scope | Duplicate groups | Duplicate rows | Excess rows | Max group size |
| :--- | :--- | ---: | ---: | ---: | ---: |
| Test | Within-subject | 27 | 54 | 27 | 2 |
| Test | Cross-subject | 78 | 156 | 78 | 2 |
| Dev | Within-subject | 1 | 2 | 1 | 2 |
| Val | Cross-subject | 2 | 4 | 2 | 2 |

On the test set, full matches question_options exactly (same 108 groups / 108 excess rows). A separate label-conflict scan (same stem + options, **different** ground-truth labels) found **no** inconsistent items.

### Test Within-Subject Item Duplicates (by subject)

| Subject | Groups | Excess rows |
| :--- | ---: | ---: |
| college_physics | 11 | 11 |
| high_school_psychology | 11 | 11 |
| public_relations | 2 | 2 |
| elementary_mathematics | 1 | 1 |
| professional_psychology | 1 | 1 |
| us_foreign_policy | 1 | 1 |
| **Total** | **27** | **27** |

### Example Cases

**Same item within a subject** (college_physics_test.csv):

> *White light is normally incident on a puddle of water… A thin (500 nm) layer of oil…*

- Rows **2** and **98**, identical options, label **C** in both copies.

**Same item across subjects** (test):

> *Kinase reactions:*

- clinical_knowledge_test.csv line **35** (label B)
- college_medicine_test.csv line **145** (label B)

**Stem match that is *not* the same item** (astronomy_test.csv) — illustrates why stem-only screening over-counts:

> *Why is Saturn almost as big as Jupiter despite its smaller mass?*

- Rows **43** and **51** share the stem and options A/B, but **C** and **D** differ (option order / distractors changed). Counted under question, not under question_options.

### Key Observations:
- Moving from stem-only to stem+options on the **test** set cuts within-subject groups from **78 → 27**. Many “duplicates” are the same wording with different distractors — soft repeats, not identical items.
- Cross-subject repeats barely change (**80 → 78** groups): most shared stems across subjects are true same-item copies (same A–D text).
- On test, **78 excess cross-subject rows** plus **27 excess within-subject rows** are exact item repeats under question_options. That is a measurable inflation of the evaluation set if every row is scored independently.
- Dev has one exact duplicate pair in college_physics; val has two cross-subject item pairs.
- No same-item / conflicting-label cases were found — when the item repeats, the ground-truth letter agrees.

### Potential Consequences
- Models that memorize or cache by stem may get **double credit** for one underlying item.
- Cross-subject leakage (e.g. clinical ↔ medicine) can make category-level scores less independent than they appear.
- Cleaning decisions should use **question_options**, not stem-only counts; stem-only remains useful as a diagnostic for near-duplicates.

## Category and Subject-Level Bias Patterns
By mapping the 57 subjects into four broad academic divisions (STEM, Humanities, Social Sciences, and Applied/Professional), highly localized patterns of bias were discovered:

| Academic Category | Total Questions | Longest Option Bias (%) | Negation/Except Questions (%) |
| :--- | :--- | :--- | :--- |
| **STEM** | 3,410 | **45.19%** | 7.21% |
| **Other (Applied)** | 4,765 | **33.87%** | 21.24% |
| **Humanities** | 3,145 | **32.46%** | **44.07%** |
| **Social Sciences** | 2,722 | **31.81%** | 7.49% |

### Key Observations:
- **Severe STEM Length Bias**: Correct answers in STEM subjects are extremely detailed (e.g. longer formulas or detailed step-by-step options). 
   - **global_facts**: In the Test set, the correct option is the longest **72.0%** of the time.
   - **elementary_mathematics**: The correct option is the longest **64.02%** of the time.
- **Humanities Negation Overload**: Nearly half (**44.07%**) of all Humanities questions contain negation words (*"not"*, *"except"*, *"false"*). LLMs might struggle disproportionately with negative constraint reasoning, meaning Humanities scores might get influenced by this linguistic structure.

## Analysis Scripts
All this information has been extracted using the following scripts -
1. **[analyze_dataset.py](../experiments/analyze_dataset.py)**: Performs aggregate stats (class distribution, option length bias, duplicates, and anomaly count).
2. **[find_specific_anomalies.py](../experiments/find_specific_anomalies.py)**: Prints the exact file names, line numbers, and contents of the questions containing missing values, identical options, or leakage.
3. **[subject_bias_analysis.py](../experiments/subject_bias_analysis.py)**: Segments findings by academic categories and evaluates negation frequency.
4. **[answer_position_bias.py](../experiments/answer_position_bias.py)**: Checks the skewness towards a particular option per subject.
5. **[none_all_above_analysis.py](../experiments/none_all_above_analysis.py)**: Checks the presence of catch-all options and their correctness when present.
6. **[find_duplicate_questions.py](../experiments/find_duplicate_questions.py)**: Finds repeated question stems / items within and across subjects (`--match question` screening vs `question_options` primary); writes detail, summary, and label-conflict CSVs under `results/`.