"""
env.py

Load a project-root `.env` into os.environ without an external dependency.

Walks upward from a starting directory until it finds `.env`, then parses
simple KEY=VALUE lines (supports optional quotes; ignores blank/# comments).

No CLI arguments — library helper only.

Typical keys used by utilities.llm
----------------------------------
  GROQ_API_KEY / GROQ_API_KEY_1 ...
  OPENAI_API_KEY / OPENAI_API_KEY_1 ...
  ANTHROPIC_API_KEY / ANTHROPIC_API_KEY_1 ...
  GEMINI_API_KEY or GOOGLE_API_KEY / GEMINI_API_KEY_1 ...
"""

from __future__ import annotations

import os
from typing import Optional


def load_dotenv(start_dir: Optional[str] = None) -> Optional[str]:
    """
    Walk upward from start_dir (default: this file's package root) until a
    `.env` is found, then load KEY=VALUE pairs into os.environ.

    Returns the path of the loaded file, or None if none was found.
    """
    search_dir = start_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    while True:
        env_path = os.path.join(search_dir, ".env")
        if os.path.exists(env_path):
            with open(env_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, val = line.split("=", 1)
                        os.environ[key.strip()] = val.strip().strip('"').strip("'")
            return env_path
        parent = os.path.dirname(search_dir)
        if parent == search_dir:
            return None
        search_dir = parent
