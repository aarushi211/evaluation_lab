# MMLU Dataset Limitations & Quality Analysis

This document outlines key quality issues, biases, and structural limitations found via programmatic analysis of the MMLU dataset splits (`dev`, `val`, `test`).

---

## 1. Option Length Bias (The "Longest Option" Heuristic)
A common issue in human-designed multiple-choice questions is that the correct answer is often drafted with more detail and nuance, making it statistically longer than the incorrect distractors. 

Our analysis shows that **MMLU is heavily susceptible to option length bias**:
- **Dev Set**: The correct option is the longest **38.25%** of the time.
- **Val Set**: The correct option is the longest **34.55%** of the time.
- **Test Set**: The correct option is the longest **35.91%** of the time.

> [!NOTE]
> Since there are 4 options, a completely random guess would pick the longest option **25%** of the time. If a model simply employs a heuristic to *always predict the longest option*, it will achieve **~36% accuracy** on the test set, significantly outperforming the random chance baseline (25%).

---

## 2. Cross-Split Data Leakage
Ideally, validation and test splits should have zero overlap to ensure evaluations are clean. However, I identified **47 questions** that are leaked across splits:
- **Test vs. Val**: 47 exact question matches.
- **Test vs. Dev**: 5 exact question matches.
- **Val vs. Dev**: 1 exact question match.

### Example Leakage Cases
1. **Contractile Proteins**:
   - **Test File**: `clinical_knowledge_test.csv` (Line 15, Label: B)
   - **Val File**: `college_medicine_val.csv` (Line 11, Label: B)
   - **Question**: *"The two principal contractile proteins found in skeletal muscle are:"*
2. **Endurance Sports**:
   - **Test File**: `clinical_knowledge_test.csv` (Line 73, Label: D)
   - **Val File**: `college_medicine_val.csv` (Line 7, Label: D)
   - **Question**: *"Which of the following physiological characteristics is not important for success in endurance events such as the marath..."*

---

## 3. Formatting & Data Quality Issues

### Missing Options (NaNs) [Option is None AKA None of the above that it caught as missing option]
MMLU is standardly treated as a 4-choice benchmark (A, B, C, D). However, several questions contain missing options (representing 3-choice questions or missing data fields):
- **Test Set**: 10 questions have missing options.
- **Val Set**: 2 questions have missing options.

> [!WARNING]
> In some cases, the correct ground-truth label points to the missing option!
> For example, in [college_physics_test.csv](evaluation_lab/datasets/MMLU/data/data/test/college_physics_test.csv#L61) (Line 61), the correct label is **D**, but option D is **empty (NaN)**.

| Subject File | Line No. | Issue | Question / Label |
| :--- | :--- | :--- | :--- |
| `college_mathematics_test.csv` | 9 | Option A is missing (NaN) | *Let V be a finite-dimensional real vector space...* (Label: C) |
| `college_physics_test.csv` | 61 | Option D is missing (NaN) | *The first five harmonics... Which of the harmonics, if any, will survive...* (Label: **D**) |
| `electrical_engineering_test.csv` | 42 | Option D is missing (NaN) | *Which of the following memories uses one transistor and one capacitor...* (Label: B) |

### Identical / Duplicated Options
Some questions offer duplicate choices (e.g. Option A and Option B have identical text), which reduces the effective number of options or introduces ambiguity:
- **Test Set**: 9 occurrences
- **Val Set**: 3 occurrences
- **Dev Set**: 1 occurrence

| Subject File | Line No. | Question | Choices |
| :--- | :--- | :--- | :--- |
| `elementary_mathematics_val.csv` | 37 | *What is the value of \|3 + 5\| − \|−4\|?* | **A**: 12, **B**: -4, **C**: 4, **D**: 12 (A & D identical) |
| `business_ethics_test.csv` | 3 | *______ are the obligations of workers...* | **A**: Employee rights, **B**: Employee rights, **C**: Employer duties, **D**: Employee duties (A & B identical) |
| `high_school_chemistry_test.csv` | 141 | *When potassium perchlorate dissolves in water...* | **A**: ...spontaneous because it is exothermic, **D**: ...spontaneous because it is exothermic (A & D identical) |

---

## 4. Category and Subject-Level Bias Patterns
By mapping the 57 subjects into four broad academic divisions (STEM, Humanities, Social Sciences, and Applied/Professional), I discovered highly localized patterns of bias:

| Academic Category | Total Questions | Longest Option Bias (%) | Negation/Except Questions (%) |
| :--- | :--- | :--- | :--- |
| **STEM** | 3,410 | **45.19%** | 7.21% |
| **Other (Applied)** | 4,765 | **33.87%** | 21.24% |
| **Humanities** | 3,145 | **32.46%** | **44.07%** |
| **Social Sciences** | 2,722 | **31.81%** | 7.49% |

### Key Observations:
1. **Severe STEM Length Bias**: Correct answers in STEM subjects are extremely detailed (e.g. longer formulas or detailed step-by-step options). 
   - **`global_facts`**: In the Test set, the correct option is the longest **72.0%** of the time.
   - **`elementary_mathematics`**: The correct option is the longest **64.02%** of the time.
2. **Humanities Negation Overload**: Nearly half (**44.07%**) of all Humanities questions contain negation words (*"not"*, *"except"*, *"false"*). LLMs might struggle disproportionately with negative constraint reasoning, meaning Humanities scores might get influenced by this linguistic structure.

---

## How to Run the Analysis Scripts
I have added three Python tools in your `benchmarks/MMLU/experiments/` directory to help you run and extend these checks:
1. **[analyze_dataset.py](evaluation_lab/benchmarks/MMLU/experiments/analyze_dataset.py)**: Performs aggregate stats (class distribution, option length bias, duplicates, and anomaly count).
2. **[find_specific_anomalies.py](evaluation_lab/benchmarks/MMLU/experiments/find_specific_anomalies.py)**: Prints the exact file names, line numbers, and contents of the questions containing missing values, identical options, or leakage.
3. **[subject_bias_analysis.py](evaluation_lab/benchmarks/MMLU/experiments/subject_bias_analysis.py)**: Segments findings by academic categories and evaluates negation frequency.

