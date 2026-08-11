"""
utilities/__init__.py

Shared evaluation toolkit for the Evaluation Lab.

Import from here in benchmark scripts so provider / parsing / checkpointing
logic does not get reimplemented per paper (MMLU, HELM, TruthfulQA, …).

Public exports
--------------
  load_dotenv, project_root, ensure_project_on_path, dataset_dir, resolve_data_dir
  LLMEvaluator, SUPPORTED_PROVIDERS
  format_question, generate_few_shot_prefix, extract_answer
  make_question_id, make_row_id, assign_eval_ids, load_processed_ids,
  filter_unprocessed, append_result_row
  add_llm_args, add_data_dir_arg, add_workers_arg
  run_parallel

Shared CLI flags
----------------
  --provider   {ollama, groq, openai, anthropic, gemini}  (default: ollama)
  --model      Model name / id                            (default: llama3.2)
  --api_key    API key; comma-separated for rotation      (optional; else env)
  --json       Request JSON-shaped A/B/C/D answers        (flag)
  --data_dir   Path to folder containing test/ (dev/, val/); Colab-friendly
  --workers    Concurrent API threads (default: 1; try 4–8 for cloud APIs)
"""

from utilities.env import load_dotenv
from utilities.paths import (
    project_root,
    ensure_project_on_path,
    dataset_dir,
    resolve_data_dir,
)
from utilities.llm import LLMEvaluator, SUPPORTED_PROVIDERS
from utilities.mcq import format_question, generate_few_shot_prefix, extract_answer
from utilities.checkpoint import (  # noqa: E501
    make_question_id,
    make_row_id,
    assign_eval_ids,
    load_processed_ids,
    filter_unprocessed,
    append_result_row,
)
from utilities.cli import add_llm_args, add_data_dir_arg, add_workers_arg
from utilities.parallel import run_parallel

__all__ = [
    "load_dotenv",
    "project_root",
    "ensure_project_on_path",
    "dataset_dir",
    "resolve_data_dir",
    "LLMEvaluator",
    "SUPPORTED_PROVIDERS",
    "format_question",
    "generate_few_shot_prefix",
    "extract_answer",
    "make_question_id",
    "make_row_id",
    "assign_eval_ids",
    "load_processed_ids",
    "filter_unprocessed",
    "append_result_row",
    "add_llm_args",
    "add_data_dir_arg",
    "add_workers_arg",
    "run_parallel",
]
