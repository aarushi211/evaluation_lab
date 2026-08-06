"""
cli.py

Shared argparse helpers so every eval script exposes the same LLM flags.

add_llm_args(parser, default_provider="ollama", default_model="llama3.2")
registers:

  --provider   {ollama, groq, openai, anthropic, gemini}  (default: ollama)
  --model      Model name / id                            (default: llama3.2)
  --api_key    API key; comma-separated for key rotation  (optional)
  --json       Request JSON-shaped A/B/C/D answers        (flag)

add_data_dir_arg(parser) registers:

  --data_dir   Path to the dataset folder that contains test/ (and optionally
               dev/, val/). Default: <repo>/datasets/MMLU/data/data
               Useful in Colab / notebooks where the data lives elsewhere.

Scripts typically call add_llm_args(parser) / add_data_dir_arg(parser) then
add their own flags.
"""

from __future__ import annotations

import argparse

from utilities.llm import SUPPORTED_PROVIDERS


def add_llm_args(
    parser: argparse.ArgumentParser,
    *,
    default_provider: str = "ollama",
    default_model: str = "llama3.2",
) -> argparse.ArgumentParser:
    """Add --provider / --model / --api_key / --json flags used across evals."""
    parser.add_argument(
        "--provider",
        type=str,
        default=default_provider,
        choices=list(SUPPORTED_PROVIDERS),
        help="API provider (ollama, groq, openai, anthropic, gemini)",
    )
    parser.add_argument("--model", type=str, default=default_model, help="Model name / id")
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="API key (comma-separated for rotation). Falls back to provider env vars.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Request JSON answers when the provider supports it",
    )
    return parser


def add_data_dir_arg(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add --data_dir pointing at the folder that contains test/ (and usually
    dev/, val/). Defaults to None so callers fall back via resolve_data_dir().
    """
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help=(
            "Path to the MMLU data folder containing test/ (and optionally "
            "dev/, val/). Defaults to <repo>/datasets/MMLU/data/data. "
            "Pass this explicitly in Colab, e.g. "
            "--data_dir /content/data"
        ),
    )
    return parser
