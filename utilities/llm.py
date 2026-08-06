"""
llm.py

Multi-provider LLM client for evaluation scripts (LLMEvaluator).

Supported providers
-------------------
  ollama      Local generate API (default http://localhost:11434/api/generate)
  groq        OpenAI-compatible chat completions
  openai      OpenAI chat completions
  anthropic   Anthropic Messages API
  gemini      Google Gemini via OpenAI-compatible endpoint

API keys (--api_key or env; comma-separated / numbered suffixes for rotation)
---------------------------------------------------------------------------
  GROQ_API_KEY / GROQ_API_KEY_1 ...
  OPENAI_API_KEY / OPENAI_API_KEY_1 ...
  ANTHROPIC_API_KEY / ANTHROPIC_API_KEY_1 ...
  GEMINI_API_KEY or GOOGLE_API_KEY / GEMINI_API_KEY_1 ...

LLMEvaluator constructor arguments
----------------------------------
  provider      One of SUPPORTED_PROVIDERS
  model_name    Provider model id (e.g. llama3.2, gpt-4o-mini, gemini-2.0-flash)
  api_key       Optional explicit key(s); else env vars above
  api_url       Optional override of the default endpoint URL
  use_json      If True, ask for {"answer": "A"|"B"|"C"|"D"} when supported
  max_tokens    Completion budget (default 20; anthropic uses at least 32)
  temperature   Sampling temperature (default 0.0)
  timeout       HTTP timeout in seconds (default 60)

Key rotation + exponential backoff on HTTP 429 is shared across cloud providers.
This module is a library — CLI flags live in utilities.cli.add_llm_args.
"""

from __future__ import annotations

import os
import random
import time
from typing import Dict, List, Optional

import requests

from utilities.env import load_dotenv

# Load .env once on import so scripts don't each reimplement it.
_env_path = load_dotenv()
if _env_path:
    print(f"Loaded .env from: {_env_path}")

JSON_SYSTEM_PROMPT = (
    "You are a multiple choice question evaluator. You must return a JSON object "
    "with the key 'answer' containing the correct option letter (A, B, C, or D) only."
)
JSON_USER_SUFFIX = '\nReturn your selection in JSON: {"answer": "A"} (or B, C, D).'

SUPPORTED_PROVIDERS = ("ollama", "groq", "openai", "anthropic", "gemini")

# OpenAI-compatible chat completion endpoints
_OPENAI_COMPAT: Dict[str, Dict[str, object]] = {
    "groq": {
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "env_keys": ("GROQ_API_KEY",),
        "supports_json_mode": True,
    },
    "openai": {
        "url": "https://api.openai.com/v1/chat/completions",
        "env_keys": ("OPENAI_API_KEY",),
        "supports_json_mode": True,
    },
    "gemini": {
        # Google's OpenAI-compatible Gemini endpoint
        "url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "env_keys": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        "supports_json_mode": True,
    },
}


def collect_api_keys(
    env_key_names: tuple,
    explicit_key: Optional[str] = None,
) -> List[str]:
    """
    Collect API keys from an explicit string and/or environment.

    Supports comma-separated values and numbered suffixes, e.g. OPENAI_API_KEY_1.
    """
    keys: List[str] = []

    if explicit_key:
        keys.extend(k.strip() for k in explicit_key.split(",") if k.strip())

    for env_name in env_key_names:
        main = os.environ.get(env_name)
        if main:
            keys.extend(k.strip() for k in main.split(",") if k.strip())

        prefix = f"{env_name}_"
        for env_key, env_val in os.environ.items():
            if env_key.startswith(prefix) and env_val.strip():
                keys.append(env_val.strip())

    seen = set()
    return [k for k in keys if not (k in seen or seen.add(k))]


class LLMEvaluator:
    def __init__(
        self,
        provider: str,
        model_name: str,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        use_json: bool = False,
        max_tokens: int = 20,
        temperature: float = 0.0,
        timeout: int = 60,
    ):
        self.provider = provider.lower().strip()
        if self.provider not in SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Unknown provider '{provider}'. Choose from: {', '.join(SUPPORTED_PROVIDERS)}"
            )

        self.model_name = model_name
        self.api_url = api_url
        self.use_json = use_json
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

        self.api_keys: List[str] = []
        self.key_index = 0

        if self.provider == "ollama":
            if not self.api_url:
                self.api_url = "http://localhost:11434/api/generate"
        elif self.provider == "anthropic":
            self.api_keys = collect_api_keys(("ANTHROPIC_API_KEY",), api_key)
            if not self.api_keys:
                raise ValueError(
                    "ANTHROPIC_API_KEY not found in .env or environment variables."
                )
            if not self.api_url:
                self.api_url = "https://api.anthropic.com/v1/messages"
            print(f"Loaded {len(self.api_keys)} Anthropic API key(s) for rotation.")
        else:
            # OpenAI-compatible: groq / openai / gemini
            cfg = _OPENAI_COMPAT[self.provider]
            self.api_keys = collect_api_keys(cfg["env_keys"], api_key)
            if not self.api_keys:
                names = " / ".join(cfg["env_keys"])
                raise ValueError(f"{names} not found in .env or environment variables.")
            if not self.api_url:
                self.api_url = cfg["url"]
            print(
                f"Loaded {len(self.api_keys)} {self.provider} API key(s) for rotation."
            )

    def get_current_key(self) -> str:
        if not self.api_keys:
            return ""
        return self.api_keys[self.key_index]

    def rotate_key(self) -> None:
        if len(self.api_keys) > 1:
            self.key_index = (self.key_index + 1) % len(self.api_keys)
            print(f"Rotating to API key index {self.key_index}...")

    def query(self, prompt: str) -> str:
        if self.provider == "ollama":
            return self._query_ollama(prompt)
        if self.provider == "anthropic":
            return self._query_anthropic(prompt)
        return self._query_openai_compat(prompt)

    # ------------------------------------------------------------------
    # Backends
    # ------------------------------------------------------------------

    def _query_ollama(self, prompt: str) -> str:
        modified_prompt = prompt
        if self.use_json:
            modified_prompt += (
                "\nYou must output JSON format with the key 'answer' mapping "
                "to either 'A', 'B', 'C', or 'D'."
            )

        data = {
            "model": self.model_name,
            "prompt": modified_prompt,
            "options": {"temperature": self.temperature},
            "stream": False,
        }
        if self.use_json:
            data["format"] = "json"

        try:
            response = requests.post(self.api_url, json=data, timeout=self.timeout)
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except Exception as e:
            print(f"Ollama API Error: {e}. Is Ollama running?")
            return ""

    def _query_openai_compat(self, prompt: str) -> str:
        messages = []
        user_content = prompt
        if self.use_json:
            messages.append({"role": "system", "content": JSON_SYSTEM_PROMPT})
            user_content = prompt + JSON_USER_SUFFIX
        messages.append({"role": "user", "content": user_content})

        data = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        cfg = _OPENAI_COMPAT[self.provider]
        if self.use_json and cfg.get("supports_json_mode"):
            data["response_format"] = {"type": "json_object"}

        return self._request_with_retries(
            url=self.api_url,
            data=data,
            headers_fn=lambda key: {
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            parse_fn=lambda res: res["choices"][0]["message"]["content"].strip(),
        )

    def _query_anthropic(self, prompt: str) -> str:
        user_content = prompt
        system = None
        if self.use_json:
            system = JSON_SYSTEM_PROMPT
            user_content = prompt + JSON_USER_SUFFIX

        data = {
            "model": self.model_name,
            "max_tokens": max(self.max_tokens, 32),
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": user_content}],
        }
        if system:
            data["system"] = system

        return self._request_with_retries(
            url=self.api_url,
            data=data,
            headers_fn=lambda key: {
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            parse_fn=lambda res: "".join(
                block.get("text", "")
                for block in res.get("content", [])
                if block.get("type") == "text"
            ).strip(),
        )

    def _request_with_retries(
        self,
        url: str,
        data: dict,
        headers_fn,
        parse_fn,
        max_retries: int = 8,
        base_delay: float = 2.0,
    ) -> str:
        for attempt in range(max_retries):
            current_key = self.get_current_key()
            headers = headers_fn(current_key)
            try:
                response = requests.post(
                    url, json=data, headers=headers, timeout=self.timeout
                )

                if response.status_code == 429:
                    print(f"[429] Rate limit hit (Attempt {attempt + 1}/{max_retries}).")
                    if len(self.api_keys) > 1 and attempt < len(self.api_keys):
                        self.rotate_key()
                        continue

                    sleep_time = base_delay * (
                        2
                        ** (
                            attempt - len(self.api_keys) + 1
                            if len(self.api_keys) > 1
                            else attempt
                        )
                    )
                    sleep_time += random.uniform(0.5, 1.5)
                    print(f"Sleeping for {sleep_time:.2f} seconds before retrying...")
                    time.sleep(sleep_time)
                    self.rotate_key()
                    continue

                response.raise_for_status()
                return parse_fn(response.json())

            except requests.exceptions.RequestException as e:
                print(f"Request exception (Attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    self.rotate_key()
                    time.sleep(base_delay)
                else:
                    print("Max retries reached. Returning empty response.")
                    return ""
        return ""
