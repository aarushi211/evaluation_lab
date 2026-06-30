# MEASURING MASSIVE MULTITASK LANGUAGE UNDERSTANDING
### Assumption
- To measure the knowledge of the model on various subjects accuracy is used as the only metric.
- Proffessional exams test knowledge and not memorization.
- The benchmark assumes accuracy reflects knowledge rather than chance or test-taking strategies.
- MCQs represent language understanding
- Pretraining contains sufficient knowledge
- Questions are representative of each subject.

### Possible Weaknesses
- Multiple choice encourages guessing.
- Knowledge not equal reasoning.
- Benchmark contamination possible.
- Only English.
- Doesn't test multimodal reasoning.
- No long-form generation.
- Accuracy treats all questions equally.
- Each subject has equal importance in the overall score.

### Questions
- Why exactly five-shot?
- Why exactly four options?
- Why not adaptive evaluation?
- Why accuracy instead of F1?
- Would CoT improve performance?
- Would today's GPT-4 still fail morality?
- Can benchmark contamination be measured?
- What is causing the lopsided performance of the models especially in subjects like morality and law. Is it the lack of information, or mix of different views from people on the internet, or authors biasness towards the topic. Everyone has different views on morality.
- How is using MCQs measuring language understanding?
- What exactly is language understanding?
- Current models use tool calling for specialized tasks including calculation, however is it possible to calibrate a language model to actually perform such tasks and is it feasible to do so?
- Why is UnifiedQA sensitive to question formatting and not GPT-3?
- If QA were on the same page in PDFs or website, would that have resulted in positive correlation between accuracy and entropy?


### My Takeaways
- Benchmarking should evolve as models improve.
- Evaluation is as important as model architecture.
- Measuring knowledge is different from measuring reasoning.
- Large models know more than small models, but knowledge is uneven.
- Calibration matters. Accuracy alone is insufficient.

### Research Questions
- How to evaluate NLG.
- Testing multimodal and multi-lingual capibilities.
- Would RAG help improve the metrics and improve overall accuracy for each subjects.