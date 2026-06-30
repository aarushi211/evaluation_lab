# MMLU paper -
1. 57 tasks - s including elementary mathematics, US history, computer science, law, and more
2. Models must possess extensive world knowledge and problem solving ability.
3. **Findings**: Most models have near random chance accuracy and GPT-3 (latest) improves over random chance by 20% on avg.
4. Models have lopsided performance and do not know when they are wrong. And have near random accuracy on socially imp subjects like morality and law. 

#### Problem
1. Dissconnect b/w current benchmarks and actual capibilities of models.
2. Current benchmarks like GLUE (General Language Understanding Evaluation), SuperGLUE measure linguistic skills and not overall language understanding and hence models were quickly able to achieve these benchmarks.
3. Commonsense benchmarks measure basic reasoning and everday knowledge, however these benchmarks have also seen rapid progress.
> Even with all these benchmarks, the model's near human-level performance suggest that they are not capturing imp facets of language understanding.
4. Though models consume huge amt of data from various specializations, current benchmarks are not measuring how capable these models actually are at learning and applying knowledge from all these domains. 
5. MMLU instead will evaluate the model's pretrained knowledge exclusively using zero and few shot settings, where the difficulty would range from elementary level to advanced professional level and it tests both world knowledge and problem solving ability.
6. The granularity and breadth of the subjects makes the bench,ark ideal for identifying model's blind spots
7. 13B models achieve random chance performance of 25% while 175B GPT-3 reaches 43.9% accuracy.
8. Also noted that GPT-3 has almost 70% accuracy for its best subjects while near random performace for several other subjects.

### Related Work
1. Pretraining: In NLP we pretrain models on massive text corpora such that these models can even be used as knowledge base. However, no previous benchmarks have measured the knowledge of these models across many real world domains.
2. Fine-tuning is commonly used on downstream tasks however, few shot learning has made it possible to achieve competitive results without fine-tuning. 
3. Current commonsense benchmarks test at a child level.
4. NLG is difficult to evaluate and lacks standard metrics. So instead MMLU uses multiple choice questions as they are easy to evaluate.

### Dataset
- 15908 questions split into few shot, validation and test set.
- few shot has 5Q per subject, validation has 1540 questions and test set has 14079 questions. Each subjects contains 100 test examples.
- Human level accuracy varies on the test.

### Experiments
UnifiedQA uses T5 text to text backbone and is fine-tuned on proposed question answering dataset, where prediction is the class with the highest token overlap with UnifiedQA's text output.

Also fine-tuned RoBERTAa-base, ALVERT-xxlarge and GPT-2 on UnifiedQA training data and dev+val set. 

### Results
**Few Shot**<br>
1. 3 smaller GPT-3 models have near random accuracy (25%).
2. 175B GPT-3 has 43.9%
3. UnifiedQA which has 11B parameters attains 48.9%
4. UnifiedQA which has 50M parameters attains 29.3%

**Zero Shot**<br>
1. Smaller models have 25%
2. GPT-3 has 37.7%

- GPT-3 on US Foreign Policy - 69% and College Chemistry 26%
- UnifiedQA on marketing 82.5%
- GPT-3 performs acquires declerative knowledge better than procedural knowledge. Moral scenarios and professional knowledge also its weak point.
- Humans usually have more depth than breadth, while models have more breadth than depth. 

**Calibration Analysis**<br>
- Large NN are often miscalibrated, especially under distribution shift. 
- GPT-3 is uncalibrated. Its confidence is weakly related to its actual accuracy in zero shot setting, with diff reaching 24% for some subjects. 
- Another measure used is RMS. Eg. Elementary maths has a zero shot RMS of 19.4%.
- Models are somewhat calibrated in few-shot setting.

### Discussions
- **Multimodal Understanding:** GPT-3 does not incorporate multimodal understanding, however in the future benchmarks testing such capibilities should also be tested. Eg. "Turk Test" consisting of Amazon Mechanical Turk Intelligence Tasks.
- **Internet as Training:** Pretraining should be viewed as the primary learning stage. Instead of fine-tuning the model on a specific task and then evaluating the performance, the model should be evaluated on the knowledge it learned through pretraining. This avoids the model from memorizing the given benchmark while testing the model's knowledge. 
- **Model Limitations:** Current models are noticiably poor at modeling human (dis)approval, and performing calculations. Moreover, models do not match expert level performance (90%) on any subject. On further experimentation, it was also noted that just fine-tuning a model on even bigger dataset for a specialized topic did not increase the accuracy substantially. Current understanding indicates that 10x increase in model size must be accompanied by an approx 5X increase in data, which is an expensive process.

### Additional Notes from Appendix
- For GPT-3 is not as sensitive to Q formatting as UnifiedQA. GPT-3 gave similar accuracies while for UnfiedQA was highly dependend on the formatting.
- If a model memorized the exact question and ans during pretraining then they should attain high accuracy while the entropy would be low for that memorized question. 
    - However, it was noted that accuracy and entropy were not positively correlated, suggesting that test's low entropy questions do not correspond to memorized QA. Hence suggests questions were not memorized. 
    - Also noted that questions came from pdfs and websites where questions and answers were on seperate pages. 