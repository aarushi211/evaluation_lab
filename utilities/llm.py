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
  max_tokens    Completion budget (default 20; anthropic uses at least 32;
                GPT-5 / o-series auto-bump to >=2048 + reasoning_effort=minimal)
  temperature   Sampling temperature (default 0.0; omitted for GPT-5 / o-series)
  timeout       HTTP timeout in seconds (default 60)

Key rotation + exponential backoff on HTTP 429 is shared across cloud providers.
This module is a library — CLI flags live in utilities.cli.add_llm_args.
"""

from __future__ import annotations

import os
import random
import threading
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


def _format_http_error_response(response: requests.Response, limit: int = 500) -> str:
    """Create a compact error string for logging / exceptions."""
    try:
        body = response.text.strip()
    except Exception:
        body = ""

    if not body:
        try:
            body = str(response.json())
        except Exception:
            body = ""

    if body:
        body = " ".join(body.split())
        if len(body) > limit:
            body = body[:limit].rstrip() + "..."

    if body:
        return f"{response.status_code} {response.reason}: {body}"
    return f"{response.status_code} {response.reason}"


def _openai_model_compat(model_name: str) -> Dict[str, object]:
    """
    Newer OpenAI reasoning models (GPT-5 / o-series) need special request params:
      - omit temperature (only the API default is allowed)
      - use max_completion_tokens (chat) / max_output_tokens (responses)
      - set reasoning_effort low enough that thinking doesn't consume the whole budget
      - use a large enough token floor; otherwise finish_reason=length with empty content
    """
    name = model_name.lower().strip()
    if "/" in name:
        name = name.split("/", 1)[-1]

    is_reasoning = (
        name.startswith("gpt-5")
        or name.startswith("o1")
        or name.startswith("o3")
        or name.startswith("o4")
    )

    return {
        "is_reasoning": is_reasoning,
        "omit_temperature": is_reasoning,
        # MCQ answers are tiny, but reasoning tokens share this budget.
        "min_completion_tokens": 2048 if is_reasoning else 0,
        "reasoning_effort": "minimal" if is_reasoning else None,
    }


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
        self._key_lock = threading.Lock()

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
        with self._key_lock:
            if not self.api_keys:
                return ""
            return self.api_keys[self.key_index]

    def rotate_key(self) -> None:
        with self._key_lock:
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
        compat = _openai_model_compat(self.model_name)

        # GPT-5 / o-series: chat completions with low reasoning effort + large budget.
        # (A tiny max_completion_tokens budget is often entirely consumed by hidden
        # reasoning tokens, which yields HTTP 200 with empty content.)
        if self.provider == "openai" and compat["is_reasoning"]:
            text = self._query_openai_reasoning_chat(prompt, compat)
            if text:
                return text
            print(
                "OpenAI reasoning model returned empty content via Chat Completions; "
                "retrying via Responses API..."
            )
            return self._query_openai_responses(prompt, compat)

        # Legacy OpenAI-compatible chat-completions path (groq / gemini / older openai).
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
            "max_tokens": max(self.max_tokens, int(compat["min_completion_tokens"]) or self.max_tokens),
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
            parse_fn=self._parse_openai_chat_response,
        )

    def _query_openai_reasoning_chat(self, prompt: str, compat: Dict[str, object]) -> str:
        """Chat Completions path tuned for GPT-5 / o-series reasoning models."""
        messages = []
        user_content = prompt
        if self.use_json:
            messages.append({"role": "system", "content": JSON_SYSTEM_PROMPT})
            user_content = prompt + JSON_USER_SUFFIX
        messages.append({"role": "user", "content": user_content})

        token_budget = max(self.max_tokens, int(compat["min_completion_tokens"]))
        data = {
            "model": self.model_name,
            "messages": messages,
            "max_completion_tokens": token_budget,
        }
        if compat.get("reasoning_effort"):
            data["reasoning_effort"] = compat["reasoning_effort"]
        if self.use_json:
            data["response_format"] = {"type": "json_object"}
        # temperature intentionally omitted

        def _parse_and_warn(res: dict) -> str:
            text = self._parse_openai_chat_response(res)
            if not text:
                choice = (res.get("choices") or [{}])[0] if isinstance(res, dict) else {}
                finish = choice.get("finish_reason") if isinstance(choice, dict) else None
                usage = res.get("usage") if isinstance(res, dict) else None
                print(
                    f"Warning: empty OpenAI chat content "
                    f"(finish_reason={finish!r}, usage={usage}). "
                    f"Reasoning likely exhausted max_completion_tokens={token_budget}."
                )
            return text

        return self._request_with_retries(
            url=self.api_url or "https://api.openai.com/v1/chat/completions",
            data=data,
            headers_fn=lambda key: {
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            parse_fn=_parse_and_warn,
        )

    def _parse_openai_chat_response(self, res: dict) -> str:
        """Extract assistant text from chat-completions style payloads."""
        if not isinstance(res, dict):
            return ""

        choices = res.get("choices")
        if isinstance(choices, list):
            for choice in choices:
                if not isinstance(choice, dict):
                    continue
                message = choice.get("message")
                if isinstance(message, dict):
                    content = message.get("content")
                    if isinstance(content, str) and content.strip():
                        return content.strip()
                    if isinstance(content, list):
                        parts = []
                        for part in content:
                            if isinstance(part, dict) and part.get("type") in (
                                "text",
                                "output_text",
                            ):
                                parts.append(part.get("text", ""))
                            elif isinstance(part, str):
                                parts.append(part)
                        text = "".join(parts).strip()
                        if text:
                            return text
                    # Some SDKs put refusal text here instead of content.
                    refusal = message.get("refusal")
                    if isinstance(refusal, str) and refusal.strip():
                        return refusal.strip()
        return ""

    def _query_openai_responses(self, prompt: str, compat: Optional[Dict[str, object]] = None) -> str:
        """Responses API fallback for GPT-5 / o-series."""
        compat = compat or _openai_model_compat(self.model_name)

        if self.use_json:
            input_text = JSON_SYSTEM_PROMPT + "\n\n" + prompt + JSON_USER_SUFFIX
        else:
            input_text = prompt

        token_budget = max(self.max_tokens, int(compat["min_completion_tokens"]), 2048)
        data = {
            "model": self.model_name,
            "input": input_text,
            "max_output_tokens": token_budget,
        }
        if compat.get("reasoning_effort"):
            data["reasoning"] = {"effort": compat["reasoning_effort"]}
        if self.use_json:
            data["text"] = {"format": {"type": "json_object"}}

        def _parse_and_warn(res: dict) -> str:
            text = self._parse_openai_responses_response(res)
            if not text:
                status = res.get("status") if isinstance(res, dict) else None
                details = res.get("incomplete_details") if isinstance(res, dict) else None
                usage = res.get("usage") if isinstance(res, dict) else None
                print(
                    f"Warning: empty OpenAI Responses content "
                    f"(status={status!r}, incomplete_details={details}, usage={usage}). "
                    f"Try a higher token budget or lower reasoning effort."
                )
            return text

        return self._request_with_retries(
            url="https://api.openai.com/v1/responses",
            data=data,
            headers_fn=lambda key: {
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            parse_fn=_parse_and_warn,
        )

    def _parse_openai_responses_response(self, res: dict) -> str:
        """
        Extract assistant text from Responses API payloads.
        """
        if not isinstance(res, dict):
            return ""

        # Fast path used by OpenAI's quickstart.
        output_text = res.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()

        # Fallback for structured payloads.
        output = res.get("output")
        if isinstance(output, list):
            parts = []
            for item in output:
                if not isinstance(item, dict):
                    continue
                for block in item.get("content", []) if isinstance(item.get("content"), list) else []:
                    if isinstance(block, dict) and block.get("type") in ("output_text", "text"):
                        parts.append(block.get("text", ""))
                    elif isinstance(block, str):
                        parts.append(block)
            text = "".join(parts).strip()
            if text:
                return text

        return ""

    def _parse_openai_compat_response(self, res: dict) -> str:
        """Extract assistant text from chat-completions or responses-style payloads."""

        def _extract_text(obj) -> str:
            if obj is None:
                return ""
            if isinstance(obj, str):
                return obj.strip()
            if isinstance(obj, list):
                parts = []
                for part in obj:
                    text = _extract_text(part)
                    if text:
                        parts.append(text)
                return "".join(parts).strip()
            if isinstance(obj, dict):
                # Common direct text fields.
                for key in ("output_text", "text", "content"):
                    val = obj.get(key)
                    if isinstance(val, str) and val.strip():
                        return val.strip()

                # Some payloads nest assistant content blocks here.
                for key in ("output", "message", "choices", "delta"):
                    if key in obj:
                        text = _extract_text(obj.get(key))
                        if text:
                            return text
            return ""

        if not isinstance(res, dict):
            return ""

        # Top-level convenience fields used by some APIs.
        for key in ("output_text", "text"):
            val = res.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()

        # Responses API style.
        text = _extract_text(res.get("output"))
        if text:
            return text

        # Chat Completions style.
        choices = res.get("choices")
        if isinstance(choices, list):
            for choice in choices:
                if not isinstance(choice, dict):
                    continue
                message = choice.get("message")
                if isinstance(message, dict):
                    text = _extract_text(message.get("content"))
                    if text:
                        return text
                    # Fallback for atypical message payloads.
                    text = _extract_text(message)
                    if text:
                        return text

        return ""

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

                # Fail fast on non-retriable client errors (4xx except 429).
                if 400 <= response.status_code < 500:
                    message = _format_http_error_response(response)
                    raise RuntimeError(
                        f"{self.provider} API returned a non-retriable client error: {message}"
                    )

                response.raise_for_status()
                res = response.json()
                return parse_fn(res)

            except requests.exceptions.RequestException as e:
                # If we somehow got a non-HTTP request exception (timeout, connection
                # error, etc.), keep the existing retry behavior.
                print(f"Request exception (Attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    self.rotate_key()
                    time.sleep(base_delay)
                else:
                    print("Max retries reached. Returning empty response.")
                    return ""
        return ""