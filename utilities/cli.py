"""Shared argparse helpers for evaluation scripts."""

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
