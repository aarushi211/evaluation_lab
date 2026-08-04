import os
import glob
import random
import argparse
import pandas as pd
import requests
import re
import json
import time
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

# Load .env file by walking up from script location to find the project root
def _load_dotenv():
    search_dir = os.path.dirname(os.path.abspath(__file__))
    while True:
        env_path = os.path.join(search_dir, '.env')
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, val = line.split('=', 1)
                        os.environ[key.strip()] = val.strip().strip('"').strip("'")
            print(f"Loaded .env from: {env_path}")
            return
        parent = os.path.dirname(search_dir)
        if parent == search_dir:
            break  # reached filesystem root without finding .env
        search_dir = parent

_load_dotenv()

class LLMEvaluator:
    def __init__(self, provider: str, model_name: str, api_key: str = None, api_url: str = None, use_json: bool = False):
        self.provider = provider.lower()
        self.model_name = model_name
        self.api_url = api_url
        self.use_json = use_json
        
        self.api_keys = []
        self.key_index = 0

        if self.provider == 'groq':
            if api_key:
                self.api_keys = [k.strip() for k in api_key.split(",") if k.strip()]
            else:
                main_key = os.environ.get("GROQ_API_KEY")
                if main_key:
                    self.api_keys = [k.strip() for k in main_key.split(",") if k.strip()]
                
                # Check for GROQ_API_KEY_1, GROQ_API_KEY_2, etc.
                for env_key, env_val in os.environ.items():
                    if env_key.startswith("GROQ_API_KEY_") and env_val.strip():
                        self.api_keys.append(env_val.strip())
            
            # De-duplicate
            seen = set()
            self.api_keys = [x for x in self.api_keys if not (x in seen or seen.add(x))]

            if not self.api_keys:
                raise ValueError("GROQ_API_KEY not found in .env or environment variables.")
            print(f"Loaded {len(self.api_keys)} Groq API key(s) for rotation.")
            
        elif self.provider == 'ollama':
            if not self.api_url:
                self.api_url = "http://localhost:11434/api/generate"

    def get_current_key(self) -> str:
        if not self.api_keys:
            return ""
        return self.api_keys[self.key_index]

    def rotate_key(self):
        if len(self.api_keys) > 1:
            self.key_index = (self.key_index + 1) % len(self.api_keys)
            print(f"Rotating to API key index {self.key_index}...")

    def query(self, prompt: str) -> str:
        if self.provider == 'groq':
            url = "https://api.groq.com/openai/v1/chat/completions"
            
            messages = []
            if self.use_json:
                messages.append({
                    "role": "system", 
                    "content": "You are a multiple choice question evaluator. You must return a JSON object with the key 'answer' containing the correct option letter (A, B, C, or D) only."
                })
                prompt += "\nReturn your selection in JSON: {\"answer\": \"A\"} (or B, C, D)."
            messages.append({"role": "user", "content": prompt})

            data = {
                "model": self.model_name,
                "messages": messages,
                "temperature": 0.0,
                "max_tokens": 20
            }
            if self.use_json:
                data["response_format"] = {"type": "json_object"}

            max_retries = 8
            base_delay = 2.0
            
            for attempt in range(max_retries):
                current_key = self.get_current_key()
                headers = {
                    "Authorization": f"Bearer {current_key}",
                    "Content-Type": "application/json"
                }
                
                try:
                    response = requests.post(url, json=data, headers=headers, timeout=15)
                    
                    if response.status_code == 429:
                        print(f"[429] Rate limit hit (Attempt {attempt+1}/{max_retries}).")
                        
                        # Rotate key first before sleeping if we have multiple
                        if len(self.api_keys) > 1 and attempt < len(self.api_keys):
                            self.rotate_key()
                            continue
                            
                        # Exponential backoff + jitter
                        sleep_time = base_delay * (2 ** (attempt - len(self.api_keys) + 1 if len(self.api_keys) > 1 else attempt))
                        sleep_time += random.uniform(0.5, 1.5)
                        print(f"Sleeping for {sleep_time:.2f} seconds before retrying...")
                        time.sleep(sleep_time)
                        self.rotate_key()
                        continue
                        
                    response.raise_for_status()
                    res_json = response.json()
                    return res_json['choices'][0]['message']['content'].strip()
                    
                except requests.exceptions.RequestException as e:
                    print(f"Request exception (Attempt {attempt+1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        self.rotate_key()
                        time.sleep(base_delay)
                    else:
                        print("Max retries reached. Returning empty response.")
                        return ""
            return ""
            
        elif self.provider == 'ollama':
            # Querying Ollama local endpoint
            modified_prompt = prompt
            if self.use_json:
                modified_prompt += "\nYou must output JSON format with the key 'answer' mapping to either 'A', 'B', 'C', or 'D'."
            
            data = {
                "model": self.model_name,
                "prompt": modified_prompt,
                "options": {
                    "temperature": 0.0
                },
                "stream": False
            }
            if self.use_json:
                data["format"] = "json"
                
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
    cleaned = model_output.strip()
    if not cleaned:
        return "N/A"
    
    # Try parsing as JSON first
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict) and "answer" in data:
            ans = str(data["answer"]).strip().upper()
            if ans in ['A', 'B', 'C', 'D']:
                return ans
    except Exception:
        pass
        
    # If exactly A, B, C, D or with punctuation
    short_match = re.match(r'^\s*\(?([A-Da-d])\)?\.?\s*$', cleaned)
    if short_match:
        return short_match.group(1).upper()
        
    # Look for standalone uppercase A, B, C, D
    matches = re.findall(r'\b([A-D])\b', cleaned)
    if matches:
        return matches[-1]
        
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
        
        print(f"Q{idx+1}: GroundTruth={new_label} | Predicted={pred_label} | RawOutput='{raw_output.strip()}' | {'Correct' if is_correct else 'Incorrect'}")

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
    parser.add_argument('--json', action='store_true', help="Enforce JSON response format")
    
    args = parser.parse_args()
    
    # Data directory: relative to this script's location (benchmarks/MMLU/experiments/)
    # → go up 3 levels to project root, then into datasets/MMLU/data/data
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.abspath(os.path.join(_script_dir, '..', '..', '..'))
    data_dir = os.path.join(_project_root, 'datasets', 'MMLU', 'data', 'data')
    
    try:
        evaluator = LLMEvaluator(provider=args.provider, model_name=args.model, api_key=args.api_key, use_json=args.json)
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
