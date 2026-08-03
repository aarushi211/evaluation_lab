import os
import glob
import random
import argparse
import pandas as pd
import requests
from typing import List, Dict, Any

# Prompts based on standard MMLU evaluation formats
def format_question(question: str, options: List[str], include_answer: bool = False, correct_label: str = None) -> str:
    prompt = f"Question: {question}\n"
    prompt += f"A. {options[0]}\n"
    prompt += f"B. {options[1]}\n"
    prompt += f"C. {options[2]}\n"
    prompt += f"D. {options[3]}\n"
    prompt += "Answer:"
    if include_answer and correct_label:
        prompt += f" {correct_label}\n\n"
    return prompt

def generate_few_shot_prefix(dev_df: pd.DataFrame, num_shots: int = 5) -> str:
    prefix = ""
    # Select up to num_shots examples
    shots = dev_df.head(num_shots)
    for _, row in shots.iterrows():
        options = [row['A'], row['B'], row['C'], row['D']]
        prefix += format_question(row['question'], options, include_answer=True, correct_label=row['label'].strip())
    return prefix

class LLMEvaluator:
    def __init__(self, provider: str, model_name: str, api_key: str = None, api_url: str = None):
        self.provider = provider.lower()
        self.model_name = model_name
        self.api_key = api_key
        self.api_url = api_url

        if self.provider == 'groq':
            if not self.api_key:
                self.api_key = os.environ.get("GROQ_API_KEY")
            if not self.api_key:
                raise ValueError("GROQ_API_KEY must be provided or set in environment variables.")
        elif self.provider == 'ollama':
            if not self.api_url:
                self.api_url = "http://localhost:11434/api/generate"

    def query(self, prompt: str) -> str:
        if self.provider == 'groq':
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": self.model_name,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.0,
                "max_tokens": 5
            }
            try:
                response = requests.post(url, json=data, headers=headers, timeout=15)
                response.raise_for_status()
                res_json = response.json()
                return res_json['choices'][0]['message']['content'].strip()
            except Exception as e:
                print(f"Groq API Error: {e}")
                return ""
                
        elif self.provider == 'ollama':
            # Querying Ollama local endpoint
            data = {
                "model": self.model_name,
                "prompt": prompt,
                "options": {
                    "temperature": 0.0
                },
                "stream": False
            }
            try:
                response = requests.post(self.api_url, json=data, timeout=30)
                response.raise_for_status()
                res_json = response.json()
                return res_json.get('response', '').strip()
            except Exception as e:
                print(f"Ollama API Error: {e}. Is Ollama running?")
                return ""
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

def extract_answer(model_output: str) -> str:
    cleaned = model_output.strip().upper()
    if not cleaned:
        return "N/A"
    # Often models return "Answer: A" or just "A" or "A. Option Text"
    # Find the first occurrence of A, B, C, or D in the response
    for char in cleaned:
        if char in ['A', 'B', 'C', 'D']:
            return char
    return "N/A"

def run_evaluation(
    data_dir: str, 
    subject: str, 
    evaluator: LLMEvaluator, 
    num_shots: int = 0, 
    shuffle_options: bool = False,
    limit: int = None
):
    print(f"\n--- Evaluating Subject: {subject} (Shuffled Options: {shuffle_options}, Shots: {num_shots}) ---")
    
    test_file = os.path.join(data_dir, 'test', f"{subject}_test.csv")
    dev_file = os.path.join(data_dir, 'dev', f"{subject}_dev.csv")
    
    if not os.path.exists(test_file):
        print(f"Test file not found for {subject}")
        return
        
    test_df = pd.read_csv(test_file, header=None, names=['question', 'A', 'B', 'C', 'D', 'label'])
    
    few_shot_prefix = ""
    if num_shots > 0 and os.path.exists(dev_file):
        dev_df = pd.read_csv(dev_file, header=None, names=['question', 'A', 'B', 'C', 'D', 'label'])
        few_shot_prefix = generate_few_shot_prefix(dev_df, num_shots)

    if limit:
        test_df = test_df.head(limit)

    correct_count = 0
    total_count = 0
    
    for idx, row in test_df.iterrows():
        question = row['question']
        original_options = [str(row['A']), str(row['B']), str(row['C']), str(row['D'])]
        original_label = str(row['label']).strip()
        
        # Determine options and correct label
        if shuffle_options:
            # Pair options with their original index letters (0=A, 1=B, 2=C, 3=D)
            indexed_options = list(zip(['A', 'B', 'C', 'D'], original_options))
            random.seed(idx) # Keep shuffle deterministic for consistency
            random.shuffle(indexed_options)
            
            shuffled_options = [opt[1] for opt in indexed_options]
            # Find which position the original correct letter ended up in
            new_label = None
            for new_idx, (orig_letter, _) in enumerate(indexed_options):
                if orig_letter == original_label:
                    new_label = ['A', 'B', 'C', 'D'][new_idx]
                    break
        else:
            shuffled_options = original_options
            new_label = original_label
            
        # Build prompt
        prompt = few_shot_prefix + format_question(question, shuffled_options)
        
        # Get prediction
        raw_output = evaluator.query(prompt)
        pred_label = extract_answer(raw_output)
        
        is_correct = (pred_label == new_label)
        if is_correct:
            correct_count += 1
        total_count += 1
        
        print(f"Q{idx+1}: GroundTruth={new_label} | Predicted={pred_label} | {'Correct' if is_correct else 'Incorrect'}")

    accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
    print(f"Accuracy: {accuracy:.2f}% ({correct_count}/{total_count})")
    return accuracy

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate MMLU subjects using Groq or Ollama")
    parser.add_argument('--provider', type=str, default='ollama', choices=['ollama', 'groq'], help="API provider")
    parser.add_argument('--model', type=str, default='llama3.2', help="Model name to evaluate")
    parser.add_argument('--subject', type=str, default='anatomy', help="MMLU subject to run")
    parser.add_argument('--shots', type=int, default=0, help="Number of few-shot examples")
    parser.add_argument('--shuffle', action='store_true', help="Shuffle options to test position/length sensitivity")
    parser.add_argument('--limit', type=int, default=10, help="Limit number of questions evaluated")
    parser.add_argument('--api_key', type=str, default=None, help="Groq API key")
    
    args = parser.parse_args()
    
    data_dir = r"c:\Users\aarus\Desktop\College\Projects\evaluation_lab\datasets\MMLU\data\data"
    
    try:
        evaluator = LLMEvaluator(provider=args.provider, model_name=args.model, api_key=args.api_key)
        run_evaluation(
            data_dir=data_dir,
            subject=args.subject,
            evaluator=evaluator,
            num_shots=args.shots,
            shuffle_options=args.shuffle,
            limit=args.limit
        )
    except Exception as e:
        print(f"Initialization/Execution error: {e}")
