"""
Shared evaluation utilities for the Evaluation Lab.

Import from here in benchmark scripts so provider / parsing / checkpointing
logic does not get reimplemented per paper.
"""

from utilities.env import load_dotenv
from utilities.paths import project_root, ensure_project_on_path, dataset_dir
from utilities.llm import LLMEvaluator, SUPPORTED_PROVIDERS
from utilities.mcq import format_question, generate_few_shot_prefix, extract_answer
from utilities.checkpoint import make_question_id, load_processed_ids, append_result_row
from utilities.cli import add_llm_args

__all__ = [
    "load_dotenv",
    "project_root",
    "ensure_project_on_path",
    "dataset_dir",
    "LLMEvaluator",
    "SUPPORTED_PROVIDERS",
    "format_question",
    "generate_few_shot_prefix",
    "extract_answer",
    "make_question_id",
    "load_processed_ids",
    "append_result_row",
    "add_llm_args",
]
