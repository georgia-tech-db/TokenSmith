"""
src/benchmark_eval/judge_client.py

Unified judge call abstraction supporting two backends:

  local       — TokenSmith's local Qwen GGUF model via llama_cpp
                Uses Qwen 2.5 chat template with JSON output priming.
                Model path taken from RAGConfig.gen_model.

  openrouter  — Any model via OpenRouter API.
                Uses standard messages format.
                Model specified via --judge-model CLI flag.

Both backends use the same prompts and return parsed JSON dicts.
"""

from __future__ import annotations

import json
import re
import sys
import pathlib
from typing import Optional

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# JSON parsing — shared by both backends
# ─────────────────────────────────────────────────────────────────────────────

def parse_judge_json(raw: Optional[str]) -> Optional[dict]:
    """
    Parse JSON from judge model output.
    Handles markdown fences, partial output, and leading/trailing noise.
    """
    if not raw:
        return None

    cleaned = re.sub(
        r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE
    ).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass

    # Try completing truncated JSON
    try:
        open_b = cleaned.count("{") - cleaned.count("}")
        open_s = cleaned.count("[") - cleaned.count("]")
        completed = cleaned + ("]" * max(0, open_s)) + ("}" * max(0, open_b))
        return json.loads(completed)
    except Exception:
        pass

    print(f"    [JUDGE] JSON parse failed. Raw (first 200): {(raw or '')[:200]}")
    return None


def safe_verdict(parsed: Optional[dict], key: str,
                 valid: set, default: str) -> str:
    if not parsed:
        return default
    val = str(parsed.get(key, default)).lower().strip()
    return val if val in valid else default


# ─────────────────────────────────────────────────────────────────────────────
# Judge client
# ─────────────────────────────────────────────────────────────────────────────

JUDGE_SYSTEM = (
    "You are a precise, impartial evaluator for a question-answering system. "
    "You always respond with valid JSON and nothing else. "
    "Do not include any prose, markdown fences, or explanations outside the JSON object."
)

DEFAULT_JUDGE_MAX_TOKENS = 600


class JudgeClient:
    """
    Unified judge call interface.

    Parameters
    ----------
    backend         : "local" or "openrouter"
    local_model_path: path to GGUF model (required for local backend)
    openrouter_key  : OpenRouter API key (required for openrouter backend)
    openrouter_model: OpenRouter model ID e.g. "qwen/qwen-2.5-72b-instruct"
    """

    def __init__(
        self,
        backend:          str,
        local_model_path: str = "",
        openrouter_key:   str = "",
        openrouter_model: str = "qwen/qwen-2.5-72b-instruct",
    ) -> None:
        self.backend          = backend
        self.local_model_path = local_model_path
        self.openrouter_key   = openrouter_key
        self.openrouter_model = openrouter_model

        if backend == "local" and not local_model_path:
            raise ValueError("local backend requires local_model_path")
        if backend == "openrouter" and not openrouter_key:
            raise ValueError("openrouter backend requires openrouter_key")

    def call(
        self,
        user_prompt: str,
        max_tokens:  int = DEFAULT_JUDGE_MAX_TOKENS,
    ) -> Optional[dict]:
        """
        Run one judge call and return a parsed dict.
        Returns None if the call fails or output cannot be parsed.
        """
        if self.backend == "local":
            raw = self._call_local(user_prompt, max_tokens)
        else:
            raw = self._call_openrouter(user_prompt, max_tokens)

        return parse_judge_json(raw)

    # ── Local backend ─────────────────────────────────────────────────────────

    def _call_local(self, user_prompt: str, max_tokens: int) -> Optional[str]:
        """
        Call local Qwen GGUF via llama_cpp.
        Uses Qwen 2.5 chat template and primes output with { to force JSON.
        """
        try:
            from src.generator import run_llama_cpp
        except ImportError as exc:
            print(f"    [JUDGE] Cannot import run_llama_cpp: {exc}")
            return None

        prompt = (
            f"<|im_start|>system\n{JUDGE_SYSTEM}\n<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}\n<|im_end|>\n"
            f"<|im_start|>assistant\n{{"
        )
        try:
            result = run_llama_cpp(
                prompt=prompt,
                model_path=self.local_model_path,
                max_tokens=max_tokens,
                temperature=0.0,
            )
            text = result["choices"][0]["text"].strip()
            return "{" + text
        except Exception as exc:
            print(f"    [JUDGE] Local call failed: {exc}")
            return None

    # ── OpenRouter backend ────────────────────────────────────────────────────

    def _call_openrouter(self, user_prompt: str, max_tokens: int) -> Optional[str]:
        """
        Call a model via OpenRouter using standard messages format.
        """
        try:
            from src.llm_benchmark_generation.llm_client import call_llm
        except ImportError as exc:
            print(f"    [JUDGE] Cannot import call_llm: {exc}")
            return None

        messages = [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user",   "content": user_prompt},
        ]
        text, _ = call_llm(
            messages=messages,
            model=self.openrouter_model,
            api_key=self.openrouter_key,
            max_tokens=max_tokens,
            temperature=0.0,
        )
        return text

    # ── Description ───────────────────────────────────────────────────────────

    def describe(self) -> str:
        if self.backend == "local":
            return f"local ({pathlib.Path(self.local_model_path).name})"
        return f"openrouter ({self.openrouter_model})"