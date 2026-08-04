# Dataset Directory Structure
### 1. Core Split (`data\`)
There are 57 csv files in `dev`, `test`, and `val` corresponding to 57 distinct subjects mentioned in the benchmark. Each csv contains MCQs specific to that subject 
- `test`: This is the primary evaluation set which the researchers have used when reporting the scores.
- `val`: Small validation set used for hyperparameter tuning or prompt validation to avoid overfitting to the test dataset.
- `dev`: Used for few shot prompting. Contains exactly 5 examples per subject. 
- `auxiliarr_train`: This contains broader, larger-scale training data grouped by overarching categories (like STEM, Humanities, Social Sciences) rather than the 57 granular subjects. It's used if someone wants to actually fine-tune a model on MMLU-like data.

### 2. Legacy Text Files (`devdata` and `fulldata`)
These folders typically represent raw or concatenated versions of the data, often formatted for specific early ingestion scripts used by the original paper's authors.
- `devdata` (5 txt): Often groups the development data by broader disciplines.
- `fulldata` (74 txt): Includes a mix of all subjects and auxiliary data combined into flat text formats for easy tokenization or quick text-parsing.

---

## 3. Model Evaluation Experiments & Observations
We evaluated two models—**Groq Llama-3.1-8B-Instant** and **Local Llama-3.2 (3B)**—on the `global_facts` subject (100 questions) to test shuffling sensitivity (Position Bias) and few-shot prompting:

| Model | Setting | 0-Shot Accuracy | 0-Shot (Shuffled) | 5-Shot Accuracy | 5-Shot (Shuffled) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Llama-3.1-8B (Groq)** | JSON Mode | **40.00%** | **37.00%** | **—** | **—** |
| **Llama-3.2 (Local)** | JSON Mode | **35.00%** | **29.00%** | **31.00%** | **32.00%** |

### Key Experimental Insights:
1. **Option Shuffling Drops Performance**: Shuffling multiple-choice options breaks the length and position correlations. Llama-3.2's accuracy dropped from **35% to 29%** (near random chance), showing it relies heavily on length/position heuristics rather than pure knowledge.
2. **Few-Shot Formatting Confusion**: For smaller models like Llama-3.2, few-shot examples actually **degraded** performance and caused JSON parsing issues. Seeing multiple QA examples in context led the model to output answers to the few-shot questions instead of the target question (generating keys like `question1` through `question5` instead of answering the target question).
3. **API Key Rotation**: Groq free-tier rate limits were successfully mitigated using a custom evaluator script that rotates keys on HTTP 429 and falls back to exponential backoff with jitter.