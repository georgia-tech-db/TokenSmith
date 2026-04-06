"""
Blackboxed external LLM provider for index-time operations.

Swap between providers by changing config only — no other code changes needed:
  - "openrouter": Any model via OpenRouter API (OpenAI-compatible)
  - "gemini":     Google Gemini models via google-genai SDK
  - "local":      GGUF models via llama-cpp-python

Usage:
    from .external_llm import ExternalLLM

    llm = ExternalLLM.from_config(cfg)
    if llm is not None:
        text = llm.generate("You are helpful.", "What is 2+2?")
        data = llm.generate_json("Extract entities as JSON.", section_text)
"""

from __future__ import annotations

import json
import logging
import os
import re
from abc import ABC, abstractmethod

from .config import LlamaIndexConfig

logger = logging.getLogger(__name__)


class ExternalLLM(ABC):
    """Provider-agnostic LLM interface for index-time operations.

    Subclass and implement ``_generate_impl`` to add a new provider.
    """

    # ── public API ────────────────────────────────────────────────────────

    def generate(self, system: str, user: str) -> str:
        """Single-turn chat → plain text response."""
        return self._generate_impl(system, user, json_mode=False)

    def generate_json(self, system: str, user: str) -> dict | list:
        """Single-turn chat → parsed JSON (with one auto-retry on parse failure)."""
        json_suffix = (
            "\n\nYou MUST respond with valid JSON only. "
            "No markdown fences, no commentary outside the JSON object."
        )
        json_system = system.rstrip() + json_suffix

        raw = self._generate_impl(json_system, user, json_mode=True)
        try:
            return self._parse_json(raw)
        except ValueError:
            logger.warning("JSON parse failed, retrying with correction prompt")
            correction = (
                "Your previous response was not valid JSON. "
                "Please respond ONLY with the JSON object/array, nothing else.\n\n"
                f"Original request:\n{user}"
            )
            raw2 = self._generate_impl(json_system, correction, json_mode=True)
            return self._parse_json(raw2)

    # ── internal ──────────────────────────────────────────────────────────

    @abstractmethod
    def _generate_impl(
        self, system: str, user: str, *, json_mode: bool = False,
    ) -> str:
        """Provider-specific generation. Override in subclasses."""
        ...

    @staticmethod
    def _parse_json(text: str) -> dict | list:
        text = re.sub(r"```(?:json)?\s*", "", text)
        text = re.sub(r"```\s*$", "", text)
        text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        for pattern in [r"\{[\s\S]*\}", r"\[[\s\S]*\]"]:
            m = re.search(pattern, text)
            if m:
                try:
                    return json.loads(m.group())
                except json.JSONDecodeError:
                    continue

        raise ValueError(f"Could not parse JSON from LLM output: {text[:500]}")

    # ── factory ───────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, cfg: LlamaIndexConfig) -> ExternalLLM | None:
        """Build the right provider from config, or ``None`` if disabled."""
        provider = cfg.index_llm_provider.lower().strip()
        if provider == "none" or not cfg.index_llm_model:
            return None

        builders: dict[str, type[ExternalLLM]] = {
            "openrouter": OpenRouterLLM,
            "gemini": GeminiLLM,
            "local": LocalLLM,
        }
        if provider not in builders:
            raise ValueError(
                f"Unknown index_llm_provider: {provider!r}. "
                f"Choose from: {', '.join(builders)}"
            )
        return builders[provider].from_config(cfg)


# ── OpenRouter (OpenAI-compatible) ────────────────────────────────────────


class OpenRouterLLM(ExternalLLM):
    """OpenRouter API — works with any model they host."""

    def __init__(
        self,
        model: str,
        api_key: str,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        base_url: str = "https://openrouter.ai/api/v1",
    ):
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url.rstrip("/")

    def _generate_impl(
        self, system: str, user: str, *, json_mode: bool = False,
    ) -> str:
        import requests

        body: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}

        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=180,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    @classmethod
    def from_config(cls, cfg: LlamaIndexConfig) -> OpenRouterLLM:
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY env var required for openrouter provider"
            )
        return cls(
            model=cfg.index_llm_model,
            api_key=api_key,
            temperature=cfg.index_llm_temperature,
            max_tokens=cfg.index_llm_max_tokens,
        )


# ── Google Gemini ─────────────────────────────────────────────────────────


class GeminiLLM(ExternalLLM):
    """Google Gemini via the ``google-genai`` SDK."""

    def __init__(
        self,
        model: str,
        api_key: str,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        from google import genai

        self._client = genai.Client(api_key=api_key)

    def _generate_impl(
        self, system: str, user: str, *, json_mode: bool = False,
    ) -> str:
        from google.genai import types

        mime = "application/json" if json_mode else None
        response = self._client.models.generate_content(
            model=self.model,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
                response_mime_type=mime,
            ),
        )
        return response.text.strip()

    @classmethod
    def from_config(cls, cfg: LlamaIndexConfig) -> GeminiLLM:
        api_key = (
            os.environ.get("GEMINI_API_KEY", "")
            or os.environ.get("GOOGLE_API_KEY", "")
        )
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY or GOOGLE_API_KEY env var required for gemini provider"
            )
        return cls(
            model=cfg.index_llm_model,
            api_key=api_key,
            temperature=cfg.index_llm_temperature,
            max_tokens=cfg.index_llm_max_tokens,
        )


# ── Local GGUF via llama-cpp-python ───────────────────────────────────────


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


class LocalLLM(ExternalLLM):
    """Local GGUF model loaded with ``llama-cpp-python``."""

    def __init__(
        self,
        model_path: str,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        context_window: int = 16384,
        n_gpu_layers: int = -1,
    ):
        self.temperature = temperature
        self.max_tokens = max_tokens

        from llama_cpp import Llama

        self._llm = Llama(
            model_path=model_path,
            n_ctx=context_window,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )

    def _generate_impl(
        self, system: str, user: str, *, json_mode: bool = False,
    ) -> str:
        kwargs: dict = {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = self._llm.create_chat_completion(**kwargs)
        text = response["choices"][0]["message"]["content"].strip()
        text = _THINK_RE.sub("", text).strip()
        if "<think>" in text:
            text = text.split("</think>")[-1].strip()
        return text

    @classmethod
    def from_config(cls, cfg: LlamaIndexConfig) -> LocalLLM:
        return cls(
            model_path=cfg.index_llm_model,
            temperature=cfg.index_llm_temperature,
            max_tokens=cfg.index_llm_max_tokens,
            context_window=cfg.index_llm_context_window,
            n_gpu_layers=cfg.n_gpu_layers,
        )
