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